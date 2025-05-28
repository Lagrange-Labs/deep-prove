# QKV

## Description of the Layer
The **QKV** layer is the first layer found in each encoder/decoder of an LLM architecture, and it is employed to produce, for each word (or better, *token*) being processed by the LLM three vectors:

- A query vector `q`
- A key vector `k`
- A value vector `v`

These vectors are all computed from a vector `x` representing a token processed by the LLM. More specifically, each of this vector is obtained by multiplying `x` with corresponding weight matrices  $W_q$, $W_k$ and $W_v$, and then each resulting vector is added to a corresponding bias vector $B_q$, $B_k$ and $B_v$. Since an LLM usually needs to process more than 1 token at the same time, the computation of `q`, `k` and `v` vectors for all the input tokens is done in a "batched" fashon over matrices rather than over single vectors:

- The `s` vectors $x_i \in \mathbb{R}^m$, each representing one of the `s` input tokens being processed, are concatenated in a matrix $X \in \mathbb{R}^{s \times e}$, where the $i$-th row of $X$ is equal to $X_i$
- Define the unitary vector as $\mathbb{1}_s = 1^s$. Given the bias vectors $B_q \in \mathbb{R}^h$, $B_k \in \mathbb{R}^h$ and $B_v \in \mathbb{R}^h$, define the *bias matrices* $B_q^s \in \mathbb{R}^{s \times h}= \mathbb{1}_s*B_q^T$, $B_k^s \in \mathbb{R}^{s \times h}= \mathbb{1}_s*B_k^T$ and $B_v^s \in \mathbb{R}^{s \times h}= \mathbb{1}_s*B_v^T$; in other words, each of the $s$ rows in a bias matrix is equal to the corresponding bias vector 
- Given the weight matrices $W_q \in \mathbb{R}^{e \times h}$, $W_k \in \mathbb{R}^{e \times h}$ and $W_v \in \mathbb{R}^{e \times h}$, the output matrices $Q = XW_q + B_q^s$, $K = XW_k + B_k^s$, $V = XW_v + B_v^s$ are computed; the $i$-th row of these matrices corresponds, respectively, to the query, key and value vectors of the $i$-th input token 

The QKV layer is basically computing these three matrices `Q`, `K` and `V` from the matrix `X` representing the input tokens

## Proving the Layer
Proving the three matrix multiplications to compute `Q`, `K` and `V` can be done with a single *batched* sum-check. 

The prover starts from 3 claims $y_Q$, $y_K,$ and $y_V$ about the MLE of output matrices `Q`, `K` and `V`, respectively. More specifically, the 3 claims are computed at the random points $r_Q = (r'_Q \in \mathbb{F}^{\log(s)}, r''_Q \in \mathbb{F}^{\log(h)})$, $r_K = (r'_K \in \mathbb{F}^{\log(s)}, r''_K \in \mathbb{F}^{\log(h)})$, $r_V = (r'_V \in \mathbb{F}^{\log(s)}, r''_V \in \mathbb{F}^{\log(h)})$. Furthermore, the prover has computed the MLEs of the weight matrices $W_q$, $W_k$ and $W_v$ and the MLEs of the bias vectors $B_q$, $B_k$ and $B_v$, which were commited to in a setup phase. 

Given the 3 claims $y_Q$, $y_K,$ and $y_V$, the prover computes also the claims $B_q(r''_Q)$, $B_k(r''_K)$ and $B_v(r''_V)$, and claims $\widetilde{y_Q} = y_Q - B_q(r''_Q)$, $\widetilde{y_K} = y_K - B_k(r''_K)$, $\widetilde{y_V} = y_V - B_V(r''_V)$. The claims $B_q(r''_Q)$, $B_k(r''_K)$ and $B_v(r''_V)$ are then accumulated to be later proven with an opening proof for the corresponding committed bias polynomial.

Given the 3 claims $y'_Q$, $y'_K$, $y'_V$, the verifier samples a random challenge $\lambda$ and then the prover employs a sum-check protocol to prove the relationship:
\begin{equation}
\scriptsize
\widetilde{y_Q} + \lambda \widetilde{y_K} + \lambda^2 \widetilde{y_V} = \sum_{x \in \{0,1\}^{\log(e)}} X(r'_Q, x)*W_Q(x, r''_Q) + \lambda X(r'_K, x)*W_K(x, r''_K) + \lambda^2 X(r'_V, x)*W_V(x,r''_V)
\tag{1}
\end{equation}

Where $X$ is the MLE of the input matrix `X`. At the end of the sum-check protocol, for a random $r \in \mathbb{F}^{\log(e)}$ the following claims are produced:

- 3 claims $X(r'_Q, r)$, $X(r'_K, r)$, $X(r'_V, r)$, which can be accumulated in a single claim for the MLE polynomial $X$
- 3 Claims $y_{W_Q} = W_Q(r, r''_Q)$, $y_{W_K} = W_K(r, r''_K)$, $y_{W_V} = W_V(r, r''_V)$; each of these claims will be added to the set of claims to be opened for the corresponding committed weight polynomial

## Caching Optimization
In a LLM like GPT2, the inference is an iterative process: the same layers are evaluated multiple times, producing each time an output token. In all iterations except the first one, the input matrix $X_i$ of the QKV layer is the row-wise concatenation of the input matrix $X_{i-1}$ employed in the previous iteration and the vector $x$ representing the output token of the previous iteration. Therefore, also the `Q`, `K` and `V` matrices computed by the QKV layer are a concatenation of the corresponding matrices computed in the previous iteration and the vectors `q`, `k` and `v` computed for the new input row `x`. 

To avoid the cost of a full matrix multiplication in each iteration after the first one, it is thus possible to compute only the vectors `q`, `k` and `v` from the input row `x`, and then concatenate each vector with the corresponding matrices produced by the QKV layer in the previous iteration. 

Note that concatenation is not necessary for the matrix Q computed by the QKV layer: instead, the layer can simply outputs the vector q computed for the new input row x 

We can also port this optimization strategy to reduce the cost of proving the QKV layer in each iteration of the LLM except for the first one.   

### Proving with Caching
The proving of a QKV layer with caching is performed in 2 steps:

1. The computation of vectors `q`, `k` and `v` from the new input row `x` is proven
2. The construction of the output matrices $K \in \mathbb{R}^{(s+1) \times h}$ and $V \in \mathbb{R}^{(s+1) \times h}$ by proving concatenation of vectors $k \in \mathbb{R}^h$ and $v \in \mathbb{R}^h$ with the corresponding matrices $K_{prev} \in \mathbb{R}^{s \times h}$ and $V_{prev} \in \mathbb{R}^{s \times h}$ computed in the previous iteration. 

Since proving currently proceeds in reverse order w.r.t. inference, we start by describing how to prove step 2.

#### Prove Concatenation
We can use the accumulation strategy described [here](https://github.com/Lagrange-Labs/deep-prove/blob/master/docs/src/commitments.md#accumulation-of-different-polynomials), which, given a set of claims $y_i$, each related to the MLE of a vector $v_i$, allows to derive a single claim for the MLE of the vector $v$ given by the concatenation of the vectors $v_i$.

The proving sub-protocol starts from claims $y_K$, $y_V$ for the MLEs of the output matrices $K$ and $V$, computed at random points $r_K$ and $r_V$; note that these claims are not employed directly in the proving, but they are the claim coming from the outputs of the layer.

The prover asks for random points $r \in \mathbb{F}^{\log(h)}$, $r_{prev} \in \mathbb{F}^{\log(s) + \log(h)}$ to the verifier: 

- $r$ is used to evaluate the MLEs of the vectors $k$ and $v$ computed for the new row, obtaining the claims $y_k$ and $y_v$
- $r_{prev}$ is used to evaluate the MLEs of the matrices $K_{prev}$ and $V_{prev}$, obtaining the claims $y_{K_{prev}}$, $y_{V_{prev}}$

Now, given random challenges $a_1$, $a_2$ and $\lambda$, the verifier and the prover both computes $y_{agg} = a_1*y_{K_{prev}} + a_2*y_k + \lambda (a_1*y_{V_{prev}} + a_2*y_v)$.

Then, following the [accumulation strategy](https://github.com/Lagrange-Labs/deep-prove/blob/master/docs/src/commitments.md#accumulation-of-different-polynomials) mentioned above, the prover computes the vectors:
  
- $\mathbf{b}_{prev} \in \mathbb{F}^{s * h}$ s.t. $\mathbf{b}_{prev}[j] = \beta(j, r_{prev})$
- $\mathbf{b}_{new} \in \mathbb{F}^{h}$ s.t. $\mathbf{b}_{new}[j] = \beta(j, r)$
- $\mathbf{b} \in \mathbb{F}^{(s+1)*h}= [a_1\mathbf{b}_{prev} || a_2\mathbf{b}_{new}]$

Given the MLE $B$ of vector $\mathbf{b}$, and the MLEs $K$ and $V$ of the output matrices, the prover then proves the following relationship by sum-check
\begin{equation}
y_{agg} = \sum_{x \in \{0,1\}^{\log((s+1)h)}} B(x)(K(x) + \lambda V(x))
\tag{2} 
\end{equation}

The sum-check proof will produce the following claims at a random point $r_s$:

- Claim $B(r_s)$, which can be directly verified by the verifier
- Claims $K(r_s)$ and $V(r_s)$, which will be later accumulated with claims $y_K$ and $y_V$, respectively, to prove that they refer to the same polynomial, using the [same poly technique](https://github.com/Lagrange-Labs/deep-prove/blob/master/docs/src/commitments.md#accumulation-for-same-polynomial). 

The claims $y_k$ and $y_v$ related to the vectors $k$ and $v$ are then used to prove step 1, as shown next.

#### Prove Computation of New Vectors
A simplied variant of the sum-check employed in the general QKV layer, described in Equation (1), can be employed. The proving starts from 3 claims $y_q$, $y_k$ and $y_v$: the first one is coming directly from the output claims of the QKV layer, and it is computed over a random point $r_q \in \mathbb{F}^{\log(h)}$; the latter claims correspond to the claims $y_k$ and $y_v$ employed to prove concatenation in the same layer, and so they are evaluated over the same point $r$. 

The prover computes the claims $B_q(r_q)$, $B_k(r)$, $B_v(r)$ and generates a sum-check proof for the following relationship:
$$ y_q - B_q(r_q) + \lambda (y_k - B_k(r))  + \lambda^2 (y_v - B_v(r)) = \sum_{z \in \{0,1\}^{\log(e)}} X(z)*(W_Q(z, r_q) + \lambda W_K(z, r) + \lambda^2 W_V(z,r))
$$
where $X$ is the MLE of the input row $x$.

#### Bind Claims With Previous Iteration

Note that the claims $y_{K_{prev}}$ and $y_{V_{prev}}$, computed by the prover when proving concatentation, needs to be bound to the MLEs of the output matrices $K_{prev}$ and $V_{prev}$ computed in the previous iteration of the same QKV layer. Therefore, after the sum-check described in Equation (2), when using the same poly techique to show that claims about the output matrices $K$ and $V$ comes from the same polynomial, the prover needs to include in the same poly proof also the claims $y_{K_{prev}}$ and $y_{V_{prev}}$ computed in the sub-sequent iteration. This allows to bind the claims computed over MLEs chosen by the prover at iteration $i$ to the actual output matrices produced by the same QKV layer at iteration $i-1$.