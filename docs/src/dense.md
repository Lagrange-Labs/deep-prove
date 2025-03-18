# Dense Layer 

## Definitions

Let's define the basic operations in a dense layer:

### Matrix-Vector Multiplication

Given an input vector $x \in \mathbb{R}^n$ and a weight matrix $W \in \mathbb{R}^{m \times n}$, the matrix-vector multiplication produces an output vector $y \in \mathbb{R}^m$:

$$y = Wx$$

where:
- $x$ is the input vector of size $n$
- $W$ is the weight matrix of size $m \times n$
- $y$ is the output vector of size $m$

### Matrix-Vector with Bias

Adding a bias vector $b \in \mathbb{R}^m$ to the output gives us:

$y' = Wx + b$

where:
- $b$ is the bias vector of size $m$
- $y'$ is the final output vector of size $m$

## Proving Matrix-Vector Operations 🔍

To prove a dense layer computation in zero-knowledge, we break down the operation into steps that can be efficiently verified:

### 1. Matrix-Vector Multiplication Proof

The matrix-vector multiplication $y = Wx$ can be expressed as a sum:

$y_i = \sum_{j=1}^n W_{ij}x_j$ for each $i \in \{1,\ldots,m\}$

We prove this computation using a sumcheck protocol:

1. The verifier sends random challenges
2. The prover commits to the partial sums
3. The protocol verifies that each $y_i$ is correctly computed

### 2. Bias Addition Proof

Adding the bias is a simple element-wise addition:

$y'_i = y_i + b_i$ for each $i \in \{1,\ldots,m\}$

This operation is proved by:
1. Verifying the addition for each element
2. Ensuring the dimensions match
3. Checking the range of the output values

### Optimization Notes 💡

- The proof size is sublinear in the size of the weight matrix
- We use efficient sumcheck protocols to reduce verification time
- The prover time is optimized through parallel computation of partial sums

For more details on the cryptographic techniques used, see the [Sumcheck Protocol](./sumcheck.md) section.