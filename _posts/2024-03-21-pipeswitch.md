---
layout: post
title: "[P] PipeSwitch: Fast Pipelined Context Switching
for Deep Learning Applications"
date: 2024-03-21 17:30:00-0400
description: Notes about PipeSwitch
tags: mlsys machine-learning
categories: mlsys paper-reading
related_posts: false
---

## Motivation
- Deep learning tasks have two primary workloads: training and inference.
    - Training workloads are throughput intensive but are more flexible in latency, since it typically takes a while to finish.
    - Inference workloads are latency sensitive and have uncertain workloads.
- Typical practice is to provision separate GPU clusters for training and inference, which leads to inefficiencies
    - Inference clusters are often over-provisioned to satisfy peak workload
    - Training workloads cannot use inference clusters when inference load is low
- Ideally, unify resources and use one cluster for both tasks via time-sharing 
    - Problem: overhead for context switching is high for GPUs, which impacts inference latency requirements

## PipeSwitch
- PipeSwitch is a system proposed by [Zhihao Bai et al](https://www.usenix.org/conference/osdi20/presentation/bai) to allow efficient time-sharing of a GPU

<div class="row mt-3 justify-content-center">
    <div class="col-sm mt-3 mt-md-0" style="max-width:400px;">
        {% include figure.liquid loading="eager" path="assets/img/pipeswitch-arch.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    PipeSwitch Architecture
</div>

- PipeSwitch consists of the following subsystems:
    - Controller: Responsible for handling client requests, scheduling workers to the GPU, and directing the memory daemon to allocate and transfer model weights
    - Memory Daemon: Allocates GPU memory to the active worker and transfers model weights between host and GPU memory
    - Active/Standby workers: Active workers are workers currently executing a task in the GPU. Standby workers are workers are workers who may be idle, initializing a new task, or cleaning up a previous task

## Design Mechanisms

PipeSwitch introduces three main mechanisms that work together to reduce context switching overhead

#### Pipelined Model Transmisson
<div class="row mt-3 justify-content-center">
    <div class="col-sm mt-3 mt-md-0" style="max-width:450px;">
        {% include figure.liquid loading="eager" path="assets/img/pipeswitch-pipeline.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Pipelined model transmission can shorten latency and have better utilization of GPU resources
</div>
- Deep learning models employ a layered architecture which means that not all layer weights need to be loaded into memory to begin execution. This observation means that we can pipeline the model execution and data transmission
    - However, GPU cores executes calculations much faster than memory operations. Thus, need to consider the granularity.
        - Whole model granularity is the same as loading the whole model into memory before execution.
        - Per layer granularity operates on one layer at a time. However, this may incur significant overhead due to PCIe costs and synchronization costs.
- To this end, the authors propose *model-aware grouping* to find the optimal grouping to transmit and execute. The algorithm considers both the layer and number of layers to group together. The algorithm and proof, which I will not go into, is provided in the paper for those who are interested.

#### Unified Memory Management
- Since DL tasks require GPU memory, the authors proposed a unified memory management system to reduce overhead from allocating and transmitting the models. This is because the memory footprint of the model parameters remain unchanged during task execution. Neither forward pass nor backpropagation will change the model structure. In addition, while intermediate results are needed during training, they are produced and consumed in a structured and predictable manner. Thus, a general purpose memory management system, like what CUDA provides, is too heavyweight.

- The unified memory management has four mechanisms that helps to reduce memory overhead:
    1. **Minimize Memory Allocation Overhead**: Essentially, the memory daemon sits on top of CUDA and obtains GPU memory on startup. This eliminates the need for each task worker to allocate its own GPU memory. Instead, the daemon, having already obtained GPU memory, will pass a pointer to the worker, saving overhead. This also allows the daemon to guarantee memory isolation between workers.
    2. **Minimize memory footprint and avoid extra memory copies**: It may be a case that the same model is needed for multiple tasks. Having each task have a duplicate copy wastes memory spaces. However, having a separate process to save models in host memory would incur memory overhead from transferring the model to the task to transmit to the GPU. The memory daemon solves both as it keeps one copy of each model in host memory. Since it manages both host and GPU memory, it can also transmit the model to GPU directly. 
    3. **Minimize IPC Overhead**: I'm not too familiar with GPU IPCs and the associated overheads; though I still tried my best in understanding this mechanism. \\
    Since the model transmission is pipelined, synchronization needs to occur between the memory daemon and the worker. However, using GPU IPC is expensive. However, the authors observed a property that memory allocation for a neural network model is *deterministic*. So, given the same model and GPU memory region, the memory pointers for each pipeline group would be the same as long as the allocation order is the same between the memory daemon and the worker. Thus, we can take advantage of this and use CPU IPCs, which are cheap, in place of GPU IPCs to signal which pipeline group is transmitted.
    4. **Pin Memory**: GPUs require a page to be pinned in host memory for memory transmission. If no page is pinned in host memory, a temporary page is pinned for transmission. The authors thus pin the pages of the memory daemon to avoid such overhead.

#### Active-Standby Worker Switching
- Using separate processes for tasks incur high overhead for initialization and cleanup of the GPU environment
- Allowing multiple tasks to share one CUDA environment still incurs overhead for cleanup of the environment
- Thus, the authors propose an active-standby worker switching mechanism. 
    - Each worker is a separate process with its own CUDA context. There will be a fixed number of workers, only one of which is active at anytime.
    - All workers initialize its own CUDA context on startup.
    - If the current active worker is stopped, it needs to clean up and free GPU memory. However, since the memory is managed by the memory daemon, cleanup consists of releasing the memory pointers.
