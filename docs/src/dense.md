# Dense Layer 

## Definitions

Let's define the basic operations in a dense layer:

### Matrix-Vector Multiplication

Given an input vector $x \in \mathbb{F}^n$ and a weight matrix $W \in \mathbb{F}^{m \times n}$, the matrix-vector multiplication produces an output vector $y \in \mathbb{F}^m$:

$$y = Wx$$

where:
- $x$ is the input vector of size $n$
- $W$ is the weight matrix of size $m \times n$
- $y$ is the output vector of size $m$

### Matrix-Vector with Bias

Adding a bias vector $b \in \mathbb{F}^m$ to the output gives us:

$y' = Wx + b$

where:
- $b$ is the bias vector of size $m$
- $y'$ is the final output vector of size $m$

## Proving Dense Layer

To prove a dense layer computation in zero-knowledge, we break down the operation into steps that can be efficiently verified:

### 1. Matrix-Vector Multiplication Proof

The matrix-vector multiplication $y = Wx$ can be expressed as a sum:

$$
y_i = \sum_{j=1}^n W_{ij}x_j$ for each $i \in \{1,\ldots,m\}
$$

Remember that each layer comes with a claim about the output. In this case, the output is $y$, e.g. the claim is $(r,\tilde{y}(r))$ where $\tilde{y}(r)$ is the multilinear extension (MLE) of $y$ evaluated at point $r$.
We can express this claim as:
$$
\tilde{y}(r) = \sum_{j=1}^n \tilde{W}(r,j)\tilde{x}(j) 
$$
where $\tilde{W}(r,j)$ is the MLE of the matrix where half of its variables. It is indexed by $log_2(m)$ variables related to the rows, and $log_2(n)$ variables, the ones related to the columns. In this sumcheck, the matrix MLE has the variables related to the rows fixed at the random point $r$ coming from the claim.

This equation can be proven via a sumcheck !

The output of the sumcheck is a proof and a new claim on both $\tilde{W}(r,...)$ and $\tilde{x}(...)$. The proof is only valid if the verifier can verify those claims are too!
* Matrix claim: the matrix claim gets accumulated with our accumulation scheme setup at the beginning of the proving phase. At the end of the full flow, the accumulation procedure will output a single randomized claim from which the prover will produce a PCS opening proof. In short, we delay the verification of the claim to the end.
* Input claim: this claims now becomes the *input* claim for the previous layer in the model ! Remember deeo prove proves backwards, from the last to the first layer.

### 2. Bias Addition Proof

Note we haven't proven the bias yet. This section shows how to modify the prover and verifier to ensure the bias is taken into account:

To prove the correct computation of $y'$, the prover computes $e_b = \tilde{b}(r)$ and sends it to the verifier.

The verifier computes $e_{y} = e_y' - e_b$. Recall that $y + b = y'$ or $y = y - b$. In other words, $e_y$ is the evaluation of the MLE of $y$ at $r$, i.e. the evaluation of the output of the dense layer at $r$ !

Both parties run the matrix to vector sumcheck for proving that $y' = Wx$ as described in previous section.

The prover stores the evaluation claims of the MLEs of $b$ and $w$ so that they will be accumulated and proven at the end of the full process.

**Note**: There is a well known trick to extend the matrix by one more row to deal with the bias in a single sumcheck. Our first implementation was using it but then due to padding issues, the solution to deal with padding individually was preferred.