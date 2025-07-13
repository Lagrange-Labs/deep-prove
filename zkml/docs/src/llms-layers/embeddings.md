# Embeddings Layer

The embeddings layer implements a lookup table operation that maps discrete token identifiers to dense vector representations. This is a fundamental component in transformer-based language models.

## Mathematical Formulation

**Definition 1 (Embedding Map).** An embedding map is a function $E: \mathcal{T} \rightarrow \mathbb{R}^d$ where:
- $\mathcal{T} = \{0, 1, \ldots, v-1\}$ is the finite set of token identifiers (vocabulary)
- $v \in \mathbb{N}$ is the vocabulary size
- $d \in \mathbb{N}$ is the embedding dimension

The embedding map can be represented as a matrix $\mathbf{E} \in \mathbb{R}^{v \times d}$ where the $i$-th row $\mathbf{E}[i, :]$ contains the embedding vector for token $i$.

**Definition 2 (Embedding Layer Operation).** Given an input sequence $\mathbf{x} = (x_1, x_2, \ldots, x_s) \in \mathcal{T}^s$ of length $s$, the embedding layer computes:

$$\text{Embed}(\mathbf{x}) = (\mathbf{E}[x_1, :], \mathbf{E}[x_2, :], \ldots, \mathbf{E}[x_s, :]) \in \mathbb{R}^{s \times d}$$

## Inference Implementation

**Input:** Tensor $\mathbf{x} \in \mathbb{Z}^{s \times 1}$ containing token identifiers

**Parameters:** Embedding matrix $\mathbf{E} \in \mathbb{R}^{v \times d}$

**Output:** Tensor $\mathbf{R} \in \mathbb{R}^{s \times d}$ where $\mathbf{R}[i, :] = \mathbf{E}[x_i, :]$

The operation is equivalent to:
$$\mathbf{R} = \mathbf{H} \cdot \mathbf{E}$$

where $\mathbf{H} \in \{0,1\}^{s \times v}$ is the one-hot encoding matrix defined by:
$$\mathbf{H}[i, j] = \begin{cases} 
1 & \text{if } j = x_i \\
0 & \text{otherwise}
\end{cases}$$

## Zero-Knowledge Proof Protocol

### Protocol Overview

The goal is to prove correct execution of the embedding lookup without revealing the input tokens. We use a matrix multiplication approach that avoids committing to the large output tensor if we were using a lookup protocol approach. 

### Commitment Phase

**Commitment:** The prover commits to the embedding matrix $\mathbf{E}$ using a polynomial commitment scheme, producing commitment $C_E$.

### Proving Phase

**Step 1: One-hot Encoding Construction**
The prover constructs the one-hot encoding matrix $\mathbf{H} \in \{0,1\}^{s \times v}$ as defined above.

**Step 2: Matrix Multiplication Proof**
Using sumcheck protocol, the prover proves the relation:
$$\mathbf{R} = \mathbf{H} \cdot \mathbf{E}$$

This produces two polynomial evaluation claims:
1. **Encoding claim:** $c_H: \tilde{\mathbf{H}}(\mathbf{r}) = y_H$ for random point $\mathbf{r} \in \mathbb{F}^{\log(s) + \log(v)}$
2. **Embedding claim:** $c_E: \tilde{\mathbf{E}}(\mathbf{r}') = y_E$ for random point $\mathbf{r}' \in \mathbb{F}^{\log(v) + \log(d)}$

where $\tilde{\mathbf{H}}$ and $\tilde{\mathbf{E}}$ are the multilinear extensions of $\mathbf{H}$ and $\mathbf{E}$, respectively.

### Verification Phase

The verifier performs the following checks:

1. **Sumcheck Verification:** Verify the sumcheck proof for the matrix multiplication.
2. **Embedding Claim:** Verify $c_E$ using the polynomial commitment scheme opening.
3. **One-hot Claim:** Directly evaluate $\tilde{\mathbf{H}}(\mathbf{r})$ and verify $c_H$.

### Efficient One-hot Evaluation

**Theorem 1.** The multilinear extension of the one-hot encoding matrix $\mathbf{H}$ can be evaluated in $O(\log s + \log v)$ time.

**Proof Sketch:** The one-hot encoding has a highly structured form. For a single token ($s = 1$), we have:
$$\tilde{\mathbf{H}}(\mathbf{y}) = \beta(x_0, \mathbf{y})$$

where $\beta(a, \mathbf{y})$ is the multilinear extension of the indicator function for value $a$:
$$\beta(a, \mathbf{y}) = \prod_{i=0}^{\log v - 1} \left( a_i \cdot y_i + (1 - a_i) \cdot (1 - y_i) \right)$$

where $a_i$ is the $i$-th bit of $a$ in binary representation.

For the full sequence case, we can express the evaluation directly as:
$$\tilde{\mathbf{H}}(\mathbf{r}) = \sum_{i=0}^{s-1} \beta(i \parallel x_i, \mathbf{r})$$

where $i \parallel x_i$ denotes the concatenation of the binary representations of position $i$ and token $x_i$, and $\mathbf{r} \in \mathbb{F}^{\log s + \log v}$ is the evaluation point.

This can be computed in $O(\log s + \log v)$ time using the Lagrange interpolation formula. □

## Complexity Analysis

- **Prover Time:** $O(svd)$ for the sumcheck protocol
- **Verifier Time:** $O(\log s + \log v + \log d)$ 
- **Communication:** $O(\log(svd))$ field elements
- **Commitment Size:** $O(vd)$ for the embedding matrix