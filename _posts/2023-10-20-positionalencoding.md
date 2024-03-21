---
layout: post
title: Rotary Positional Embedding
date: 2023-10-20 17:30:00-0400
description: Notes about RoPE
tags: nlp transformer machine-learning
categories: machine-learning nlp
related_posts: false
---

## Motivation
- Position and order in sequential data are significant in understanding the data itself
    - For example, a common sentence structure in English is "subject - verb - object". So the order in which the nouns appear in relation to the verb is very important.
    - Another example is the position and order of adjectives in relation to the noun it is describing
- Past architectures, such as LSTMs and RNNs, implicitly encode positional data by continuously computing and passing along its hidden states to the next state
- Transformer models computes all the data in parallel with no mechanism that implicitly injects positional information to any data
- Thus, positional information must be applied externally

## Absolute Position Encoding
- The original Transformer paper by [Vaswani et al](https://arxiv.org/abs/1706.03762) employed absolute position encoding that is added to the vector after the embedding layer using the following formula

    \begin{align}
    PE_{(pos)} = 
    \begin{cases}
    PE_{(pos, 2i)}&=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)\\\\\\\\
    PE_{(pos, 2i+1)}&=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
    \end{cases}
    \end{align}

    where $$2i$$ is the $$2i^{th}$$ dimension, $$pos$$ denote the position of the vector, and $$d_{model}$$ is the size of embedding vector. Note that even dimensions use the sine function and odd dimensions use the cosine function.
- Each dimension corresponds to a sinusoid with wavelengths ranging from $$2\pi$$ to $$10000\cdot 2 \pi$$. Thus, in the typical usecase, it is improbable that two tokens would share the same positional encoding.
- The position encoding can also be learned, but using a function may allow the model to extrapolate positional information for sequences with lengths longer than any lengths that the model was trained on.

## Complex Numbers
The underlying principles of RoPE relies on complex algebra. To be honest, I felt a little embarrassed because I had absolutely no understanding of the paper upon the first few reads until I went back to refresh my complex numbers. 

- Complex numbers take the form of $$a + bi$$, where $$a$$ and $$b$$ are real numbers and $$i=\sqrt{-1}$$ is the imaginary unit.
- We can also represent a complex number in polar form:

    $$
    z = a + bi = r(\cos(\theta) + i\sin(\theta)) = re^{i\theta}
    $$

    where $$r= \sqrt{a^2 + b^2}$$ is called the radial component and $$\theta$$ is called the angular component.

    The latter equality is called Euler's formula: $$e^{i\theta} = \cos(\theta) + i\sin(\theta)$$
- Multiplications of two complex numbers works the same as algebraic multiplication using distributive property

    $$
    (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    $$

    We can also convert the above to matrix form:

    $$
    \begin{bmatrix}
        c & -d\\
        d & c
    \end{bmatrix}
    \begin{bmatrix}
        a \\
        b
    \end{bmatrix}
     = 
    \begin{bmatrix}
    ac - bd \\ ad + bc
    \end{bmatrix}
    $$

    Geometrically speaking, multiplication in complex domain can be thought of as an affine transformation consisting of a scale transformation and a rotation transformation. This can be seen when we replace $$c+ di$$ with its polar form $$r(cos(\theta) + i\sin(\theta))$$. The matrix form will look like:

    $$
    r\cdot
    \begin{bmatrix}
        \cos(\theta) & -\sin(\theta)\\
        \sin(\theta) & \cos(\theta)
    \end{bmatrix}
    \begin{bmatrix}
        a \\
        b
    \end{bmatrix}
     = 
    r\cdot 
    \begin{bmatrix}
    a\cos(\theta) - b\sin(\theta) \\ a\sin(\theta) + b\cos(\theta)
    \end{bmatrix}
    $$

    The above can be interpreted as scaling the complex number $$a + bi$$ by $$r$$ and then rotating it counter-clockwise by $$\theta$$. Note that the matrix is a [standard rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix) in Euclidean space.
- The inner product for the complex case is called the *Hermitian inner product.* Given $$u, v \in \mathbb{C}^n$$, it is defined as:

    $$
    \langle u, v \rangle = \sum_{i=1}^{n}u_i^*v_i
    $$

    where $$u^*$$ is the complex conjugate of $$u$$. 

## Rotary Positional Embedding
- Rotary Positional Embedding (RoPE) is a positional embedding technique proposed by [Jianlin Su et al](https://arxiv.org/abs/2104.09864).
- The main idea behind RoPE was to find a way to encode absolute positional information into a vector whilst at the same time use relative positional information during self-attention. In other words given vectors $$x_m, x_n \in \mathbb{R}^d$$, query and key functions $$f_k(\cdot), f_q(\cdot)$$, and positional information $$m, n$$, we want to find a function $$g(\cdot)$$ that satisfies the following:

    \begin{equation}
    \label{eq:relativeposition}
    \left< f_q(\mathbf{x_m}, m), f_k(\mathbf{x_n}, n)\right> = g(\mathbf{x_m}, \mathbf{x_n}, m-n)
    \end{equation}

- The solution that RoPE introduces is to map the vector $$\mathbf{x}_m \in \mathbb{R}^d$$ into $$d/2$$ subspace in the complex domain $$\mathbb{C}^{d/2}$$ by considering consecutive elements in $$\mathbf{x}_m$$ as one complex number. As in
    
    $$(x_1, x_2, \dots, x_{d-1}, x_d) \rightarrow (x_1 + i x_2 , \dots, x_{d-1} + i x_d )$$

    This necessitates that the dimension of the original vector be even, which can be done by adding an additional dimension if the vector dimension is odd. 

    We then inject positional information via rotation, which looks something like this:

    $$
    f^{RoPE}(\mathbf{x}_m, m)_{j} = (x^{(2j)} + ix^{(2j+1)})e^{mi\theta_j}
    $$

    Or in rotational matrix form:

    $$
    f^{RoPE}(\mathbf{x}_m, m)_{j} = \begin{bmatrix}
        \cos(m\theta_j) & -\sin(m\theta_j)\\
        \sin(m\theta_j) & \cos(m\theta_j)
    \end{bmatrix}
    \begin{bmatrix}
        x^{(2j)} \\
        x^{(2j+1)}
    \end{bmatrix}
    $$

    $$\theta_j$$ is the sinusoidal wave for dimension $$j$$ and is defined to be $$\theta_j = 10000^{-2j/d}$$. As a shorthand, we can let $$R_{\theta_j, m}$$ be the rotation matrix with the parameters $$\theta_j$$ and $$m$$.

    - The more general form of RoPE is given as
     
        $$
        f^{RoPE}(\mathbf{x_m}, m) = R_{\Theta, m}\mathbf{x_m}
        $$

      with the rotational matrix $$R_{\Theta, m}$$ defined as:

      $$
      R_{\Theta, m}=
      \begin{bmatrix}
      R_{\theta_0, m} & & &\\
      & R_{\theta_1, m} & \\
      & & \ddots &\\
      & & & R_{\theta_{d/2-1}, m}\\
      \end{bmatrix}
      $$

- The above formulation of RoPE can then satisfy \eqref{eq:relativeposition}. For simplicity, we show the 2D case where $$\mathbf{x_m}, \mathbf{x_n} \in \mathbb{R}^2$$: 

    $$
    \begin{align}
    \left< f_q(\mathbf{x_m}, m), f_k(\mathbf{x_n}, n)\right>_\mathbb{R} &= \text{Re}(\left< f^{RoPE}_q(\mathbf{x_m}, m), f^{RoPE}_k(\mathbf{x_n}, n)\right>_\mathbb{C})\\
        &= \text{Re}(\hat{x}^*_m e^{-im\theta} \hat{x}_ne^{in\theta})\\
        &= \text{Re}\Big(\big((x^{(1)}_mx^{(1)}_n + x^{(2)}_mx^{(2)}_n) + i(-x^{(1)}_mx^{(2)}_n + x^{(2)}_mx^{(1)}_n)\big) e^{i(n-m)\theta}\Big)\\
        &= \cos\big((n-m)\theta\big) (x^{(1)}_mx^{(1)}_n + x^{(2)}_mx^{(2)}_n) \\
        &\quad\quad - \sin\big((n-m)\theta\big)(-x^{(1)}_mx^{(2)}_n + x^{(2)}_mx^{(1)}_n)\\
        &= \mathbf{x}^\intercal_\mathbf{m} R_{\theta, n-m} \mathbf{x_n}\\
        &= g(\mathbf{x_m}, \mathbf{x_n}, n-m)
    \end{align}
    $$
    


    - A big part of my confusion stems from the fact that this equivalence is given as true: $$\mathbf{q}\mathbf{k} = \text{Re}(\mathbf{q}^* \mathbf{k})$$, which it is but I didn't know how it got there since the vectors $$\mathbf{q}, \mathbf{k}$$ are reused and the authors are relying on the readers to implicitly know if the vector is $$\mathbb{R}^d$$ or $$\mathbb{C}^{d/2}$$. If one isn't being careful and assumed both vectors are in $$\mathbb{R}^d$$, then one can observe that the complex conjugate of a real vector is itself, so $$qk = q^*k$$, which means $$qk = \text{Re}(q^*k)$$, but this isn't the point being made in these papers.