# Multi-Head Attention
We define the *multi-head attention* (MHA) layer as the step in an LLM transformer that splits the `Q`, `K` and `V` matrices produced by QKV layer in multiple heads and combines them to produce a single matrix.

## Description of the layer

Let's consider input matrices $Q \in \mathbb{R}^{s \times d}$, $K \in \mathbb{R}^{s \times d}$, $V \in \mathbb{R}^{s \times d}$, where $s$ stands for the number of tokens being processed by the attention layer and $d$ is an hyper-parameter of the model referred to as *hidden dimension*. Given a number of heads `h` defined by the model, each of these matrices is split in `h` sub-matrices over the column dimension: for instance, the matrix $Q \in \mathbb{R}^{s \times d}$ is split in `h` matrices $Q_i \in \mathbb{R}^{s \times m}$, where $m = \frac{d}{h}$, such that the matrix $Q_i$ corresponds to the $i$-th chunk of $m$ columns of matrix $Q$. Each of the sub-matrices $Q_i$, $K_i$ and $V_i$ belong to a *head* of the attention layer.

[ToDo: insert picture]

For each head, the following computation is performed.

$$
O_i = \mathtt{Softmax}(\frac{Q_i*K_i^T}{\sqrt{m}})*V_i
$$

The output matrix $O_i \in \mathbb{R}^{s \times m}$ is the output of a single head. The output of the multi-head attention is then obtained by concatenating the output $O_i$ of all the $h$ heads in a single matrix $O \in \mathbb{R}^{s \times d}$

### Masking
In LLM models like GPT-2, the attention layer is referred to as *masked self-attention* layer. We defer to [this blogpost](https://jalammar.github.io/illustrated-gpt2/) for the rationale of this design choice and its purpose. In terms of computation, the main difference in a masked self-attention layer is that, when computing the $Q_i*K_i^T$ product, the $j$-th row of $Q_i$ needs to be multiplied only with the first $j-1$ columns of matrix $K^T$, rather than with all the columns. 

In other words, the matrix $QK_i = Q_i*K_i^T \in \mathbb{R}^{s \times s}$ is a lower triangular matrix, as only the entries $QK_i[x,y]$ where $x >= y$ corresponds to an actual product between a row of matrix $Q_i$ and a column of matrix $K_i^T$. The entries $QK_i[x, y]$ with $x < y$ are instead filled with dummy values. Note that such dummy values should be big negative numbers rather than 0, as in this way these dummy values will not affect the output of the sub-sequent $\mathtt{Softmax}$ operation.

## Proving
We now describe how to prove the linear operations in multi-head attention layer, deferring how to prove softmax to another page [ToDO: insert link when available].

The fundamental building block to prove linear operations in multi-head attention layer is a *batched matrix multiplication* protocol, which allows to simultaneously prove the matrix multiplications in all the $h$ heads with a single sum-check. We start by describing how to simultaneously prove all the matrix multiplications $QK_i = Q_i*K_i^T$ for all the $h$ heads.

The input of the operation are the matrices $Q \in \mathbb{R}^{s \times h*m}$ and $K \in \mathbb{R}^{s \times h*m}$, which corresponds to the matrices produced by the QKV layer before being split in the $h$ heads. The output matrix being produce is $QK \in \mathbb{R}^{h*s \times s}$, that is we obtain a matrix where the $i$-th chunk of $s$ rows corresponds to the output matrix $Q_i*K_i^T$ computed for the $i$-th head.

We consider input matrices to be padded to the next power on each of their dimensions. That is, the matrices have shape $[\hat{s}, d]$, where $\hat{s} = 2^{\lceil \log_2(s) \rceil}$ and $d = 2^{\lceil \log_2(h*m) \rceil}$. However, the proving protocol requires also $h$ to be a power of 2, therefore we need to properly pad the columns of the input matrices $Q$ and $K$. 

### Padding
Given $\hat{h} = 2^{\lceil \log_2(h) \rceil}$ and $\hat{m} = 2^{\lceil \log_2(m) \rceil}$, the number of columns of the input matrices must become $\hat{d} = \hat{h}*\hat{m}$. This is done by padding the number of columns of each of the $h$ heads to $\hat{m}$ and by adding $\hat{h} - h$ dummy heads with $\hat{s}$ rows and $\hat{m}$ columns. This is achieved by multiplying each matrix $Q \in \mathbb{F}^{\hat{s} \times d}$, $K \in \mathbb{F}^{\hat{s} \times d}$, by the padding matrix $P \in \mathbb{F}^{d \times \hat{d}}$ defined as:
$$
P[i,j] = \begin{cases} 
    1 \quad \text{if} \ j \mod \hat{m} < m \wedge \lfloor \frac{j}{\hat{m}} \rfloor < h \wedge i = \lfloor \frac{j}{\hat{m}} \rfloor m + j \mod \hat{m} \\
    0 \quad \text{otherwise}
    \end{cases} 
$$
In other words, the padding matrix $P$ has $h*m$ non-zero entries, each yielding in the output matrix a corresponding non-zero entry in each row of the original matrices $Q$ and $K$.

After multiplying the matrices $Q$ and $K$ with $P$, we get the padded matrices $\hat{Q} \in \mathbb{F}^{\hat{s} \times \hat{h}\hat{m}}$ and $\hat{K} \in \mathbb{F}^{\hat{s} \times \hat{h}\hat{m}}$, respectively. Each of this matrix can now be split in $\hat{h}$ heads $\hat{Q}_i \in \mathbb{F}^{\hat{s} \times \hat{m}}$, $\hat{K}_i \in \mathbb{F}^{\hat{s} \times \hat{m}}$; note that: 
$$
\hat{Q}_i*\hat{K}_i^T = \begin{cases} 
    Q_i*K_i^T \quad \text{if} \quad i < h \\
    0 \quad \text{otherwise} 
    \end{cases}
$$

#### Prove Proper Padding
We rely on sum-check to prove the construction of the padded matrices $\hat{Q} \in \mathbb{F}^{\hat{s} \times \hat{d}}$ and $\hat{K} \in \mathbb{F}^{\hat{s} \times \hat{d}}$ from the matrices $Q \in \mathbb{F}^{\hat{s} \times d}$ and $K \in \mathbb{F}^{\hat{s} \times d}$, respectively. For efficiency, we also prove at the same time the construction of the padded matrix $\hat{V} \in \mathbb{F}^{\hat{s} \times \hat{d}}$ from the input matrix $V \in \mathbb{F}^{\hat{s} \times d}$, which will be later needed in the multi-head attention layer.

The proving protocol starts from the following claims about the MLEs of output matrices $\hat{Q}$, $\hat{K}$, $\hat{V}$:

- Claim $y_{\hat{Q}} = \hat{Q}(r_q^r, r_q^c)$, with $r_q^r \in \mathbb{F}^{\log_2(\hat{s})}$, $r_q^c \in \mathbb{F}^{\log_2(\hat{d})}$ being random points chosen by the verifier
- Claim $y_{\hat{K}} = \hat{K}(r_k^r, r_k^c)$, with $r_k^r \in \mathbb{F}^{\log_2(\hat{s})}$, $r_k^c \in \mathbb{F}^{\log_2(\hat{d})}$ being random points chosen by the verifier
- Claim $y_{\hat{V}} = \hat{V}(r_v^r, r_v^c)$, with $r_v^r \in \mathbb{F}^{\log_2(\hat{s})}$, $r_v^c \in \mathbb{F}^{\log_2(\hat{d})}$ being random points chosen by the verifier

Given these claims, the verifier samples random challenges $\lambda_1$, $\lambda_2$, which are shared with the prover. Then, given the padding matrix $P \in \mathbb{F}^{d \times \hat{d}}$, the proper padding of matrices $Q$, $K$ and $V$ is proven through sum-check protocol over the following relationship:
$$
y_{\hat{Q}} + \lambda_1 y_{\hat{K}} + \lambda_2 y_{\hat{V}} = \sum_{x \in \{0,1\}^{\log_2(d)}} Q(r_q^r, x)P(x, r_q^c) + \lambda_1 K(r_k^r, x)P(x, r_k^c) + \lambda_2 V(r_v^r, x)P(x, r_v^c)
$$
The sum-check protocol produces the following claims for a random point $r \in \mathbb{F}^{\log_2(d)}$:

- Claims $P(r, r_q^c)$, $P(r, r_k^c)$ and $P(r, r_k^v)$, which can be efficiently recomputed by the verifier
- Claims $y_Q = Q(r_q^r, r)$, $y_K = K(r_k^r, r)$, $y_V = V(r_v^r, r)$ which are the claims about the input matrices of the multi-head attention layer


### Prove Multi-Head Matrix Multiplication
Given the padded matrices $\hat{Q} \in \mathbb{F}^{\hat{s} \times \hat{h}\hat{m}}$ and $\hat{K} \in \mathbb{F}^{\hat{s} \times \hat{h}\hat{m}}$, we simultaneously prove the product of all the $\hat{h}$ heads $\hat{Q}_i \in \mathbb{F}^{\hat{s} \times \hat{m}}$, $\hat{K}_i \in \mathbb{F}^{\hat{s} \times \hat{m}}$ as follows. 

Consider the output matrix $\hat{QK} \in \mathbb{F}^{\hat{h}\hat{s} \times \hat{s}}$, given by the row-wise concatenation of the outputs of each of the $\hat{h}$ heads. In other words, considering this matrix as a 3d tensor (i.e., $\hat{QK} \in \mathbb{F}^{\hat{h} \times \hat{s} \times \hat{s}}$) then $\hat{QK}[i, j, k] = \hat{Q}_i*\hat{K}_i^T[j,k] $. 

The proving protocol starts from a claim $y = \hat{QK}(r_1, r_2, r_3)$, where:

- $r_1 \in \mathbb{F}^{\log(\hat{h})}$ is a random point chosen by the verifier
- $r_2 \in \mathbb{F}^{\log(\hat{s})}$ is a random point chosen by the verifier
- $r_3 \in \mathbb{F}^{\log(\hat{s})}$ is a random point chosen by the verifier

The prover now builds the vector $\mathbf{b} \in \mathbb{F}^{\hat{h}*\hat{m}}$ such that $\mathbf{b}[i] = \beta(\lfloor \frac{i}{\hat{m}} \rfloor, r_1)$. The computation of $\hat{QK}$ from matrices $\hat{Q}$ and $\hat{K}$ is then proved with a sum-check protocol over the following relationship:
\begin{equation}
y = \sum_{x \in \{0,1\}^{\log_2(\hat{h}\hat{m})}} \mathbf{b}(x) \hat{Q}(r_2, x) \hat{K}(r_3, x)
\tag{1}
\end{equation}
The sum-check protocol produces the following claims for a point $r \in \mathbb{F}^{\log_2(\hat{h}\hat{m})}$:

- Claim $\mathbf{b}(r)$, which can be recomputed locally by the verifier
- Claims $y_{\hat{Q}} = \hat{Q}(r_2, r)$ and $y_{\hat{K}} = \hat{K}(r_3, r)$, which can be used as the inputs claims in the protocol to prove the proper padding of matrices $Q$ and $K$ described [above](#prove-proper-padding) 

### Prove Masking
To apply masking on the output matrix $QK \in \mathbb{R}^{h*s \times s}$, we need to make each of the $h$ heads $QK_i \in \mathbb{R}^{s \times s}$ a lower triangular matrix, where all the entries over the diagonal are replaced with a big negative value (referred to as $\mathtt{-inf}$ henceforth).

To prove this operation, we define a *zeroifier* matrix and a *masking* matrix:

- The zeroifier matrix $\mathbf{z} \in \mathbb{R}^{s \times s}$, where $\mathbf{z}[i,j] = 0$ if $i < j$, $1$ otherwise
- The masking matrix $\mathbf{m} \in \mathbb{R}^{s \times s}$, where $m[i,j] = \mathtt{-inf}$ if $i < j$, $0$ otherwise

Given matrices $\mathbf{z}$ and $\mathbf{m}$, we can compute the masked $QK$ matrix by performing the following computation on each of the $h$ heads $QK_i$: $QK_i \otimes \mathbf{z} + \mathbf{m}$, where $\otimes$ denotes the entry-wise matrix multiplication. In a nutshell, the rationale is that we first turn all entries to be masked to 0, and then we add $\mathtt{-inf}$. The zeroing step is done to avoid potential overflows when summing the big negative number $\mathtt{-inf}$.

When proving the masking operation, we need to consider that the masking is applied over the padded matrix $\hat{QK} \in \mathbb{F}^{\hat{h} \times \hat{s} \times \hat{s}}$. In this case, we need to mask all the $\hat{s} - s$ dummy rows and dummy columns in each of the $\hat{h}$ heads $\hat{QK}_i \in \mathbb{F}^{\hat{s} \times \hat{s}}$.
Therefore, the padded zeroifier and masking matrices are computed as follows:
$$
\mathbf{z}[i,j] = \begin{cases}
1 \quad \text{if} \ i < s \wedge j < s \wedge i >= j \\
0 \quad \text{otherwise}
\end{cases}
$$
$$
\mathbf{m}[i,j] = \begin{cases}
0 \quad \qquad \text{if} \ i < s \wedge j < s \wedge i >= j \\
\mathtt{-inf} \quad \text{otherwise}
\end{cases}
$$
Since we need to apply the same matrices $\mathbf{z}$ and $\mathbf{m}$ over all the $\hat{h}$ rows of matrix $\hat{QK}\in \mathbb{F}^{\hat{h} \times \hat{s} \times \hat{s}}$, we can build corresponding matrices $\hat{\mathbf{z}}$, $\hat{\mathbf{m}} \in \mathbb{F}^{\hat{h} \times \hat{s} \times \hat{s}}$, such that $\hat{\mathbf{z}}[i,j,k] = \mathbf{z}[j,k]$ and $\hat{\mathbf{m}}[i,j,k] = \mathbf{m}[j,k]$. Note that the same relationship holds for the MLEs of the matrices.

The proving starts from a claim $y_M = \hat{QK}_M(r_1, r_2, r_3)$, where:

- $\hat{QK}_M \in \mathbb{F}^{\hat{h} \times \hat{s} \times \hat{s}}$ is the output matrix, given by the row-wise concatenation of all the $\hat{h}$ heads after masking
- $r_1 \in \mathbb{F}^{\log(\hat{h})}$ is a random point
- $r_2 \in \mathbb{F}^{\log(\hat{s})}$ is a random point
- $r_3 \in \mathbb{F}^{\log(\hat{s})}$ is a random point

Given the claim $y_M$ and the MLE of the masking polynomial $\mathbf{m}$, the prover computes the claim $y_Z = y_M - \mathbf{m}(r_2, r_3)$. Note that the claim $\mathbf{m}(r_2, r_3)$ should be efficiency recomputable by the verifier as well.

The claim $y_Z$ refers now to the *zeroified* input matrix $\hat{QK}$, so now we need to prove the *zeroification* of $\hat{QK}$. Since this operation corresponds to an entry-wise multiplication of input matrix $\hat{QK}$ with the *extended* zeroifier matrix $\hat{\mathbf{z}} \in \mathbb{F}^{\log(\hat{h} \times \hat{s} \times \hat{s})}$, we can prove this computation via the following sum-check:
$$
y_Z = \sum_{x \in \{0,1\}^{\log(\hat{h}*\hat{s}*\hat{s})}} \beta(x, r) \hat{\mathbf{z}}(x)\hat{QK}(x)   
$$
where $r$ is the random point given by concatenation of $r_1$, $r_2$ and $r_3$. The sum-check protocol produces the following claims for a random point $r_s \in \mathbb{F}^{\log(\hat{h}*\hat{s}*\hat{s})}$:

- Claim $\beta(r_s, r)$, which can be efficiency recomputed by the verifier
- Claim $\hat{\mathbf{z}}(r_s)$, which can be efficiency recomputed by the verifier: indeed, splitting $r_s$ in 3 points $r_s' \in \mathbb{F}^{\log(\hat{h})}$, $r_s'' \in \mathbb{F}^{\log(\hat{s})}$, $r_s''' \in \mathbb{F}^{\log(\hat{s})}$, we have that $\hat{\mathbf{z}}(r_s) = \hat{\mathbf{z}}(r_s', r_s'', r_s''') = \mathbf{z}(r_s'', r_s''')$, which can be efficiency recomputed by the verifier
- Claim $\hat{QK}(r_s)$. This is the claim about the unmasked matrix $\hat{QK}$, which is the input of the protocol and can later be used as claim $y$ in the sum-check protocol in Equation (1)

### Prove Final Matrix Multiplication
After applying softmax over the masked matrix $\hat{QK} \in \mathbb{F}^{\hat{h} \times \hat{s} \times \hat{s}}$, we need to prove the final matrix multiplication of multi-head attention, which is multiplying each head of $\hat{QK}$ with the corresponding head of the input matrix $\hat{V} \in \mathbb{F}^{\hat{s} \times \hat{h} \times \hat{m}}$.
More specifically, the output matrix $\hat{O} \in \mathbb{F}^{\hat{s} \times \hat{h} \times \hat{m}}$ is given by the column-wise concatenation of its $\hat{h}$ heads $\hat{O}_i \in \mathbb{F}^{\hat{s} \times \hat{m}}$, where each $\hat{O_i} = \hat{QK}_i\hat{V}_i$.

To perform this matrix multiplication, we need to consider the permuted matrix $\widetilde{V} \in \mathbb{F}^{\hat{h} \times \hat{s} \times \hat{m}}$ in place of $V$, as otherwise the matrices would have an incompatible shape when multiplied. We note that since $\hat{V}[i,j,k] = \widetilde{V}[j,i,k]$, then also for the MLEs of the matrices it holds that $\hat{V}(x_1, x_2, x_3) = \widetilde{V}(x_2,x_1,x_3)$; therefore, from a claim $y_{\widetilde{V}} = \widetilde{V}(r)$, computed over a random point $r=(r_1 \in \mathbb{F}^{\log(\hat{h})}, r_2 \in \mathbb{F}^{\log(\hat{s})}, r_3 \in \mathbb{F}^{\log(\hat{s})})$, we can get a claim about $\hat{V}$ by simply swapping coordinates of the random point $r$, i.e., $\hat{V}(r_2, r_1, r_3) = y_{\widetilde{V}}$.

To prove the multi-head matrix multiplication between $\hat{QK} \in \mathbb{F}^{\hat{h} \times \hat{s} \times \hat{s}}$ and $\widetilde{V} \in \mathbb{F}^{\hat{h} \times \hat{s} \times \hat{m}}$, we start from a claim $y_{\hat{O}} = \hat{O}(r_s, r_h, r_m)$, where:

- $r_s \in \mathbb{F}^{\log(\hat{s})}$ is a random point chosen by the verifier
- $r_h \in \mathbb{F}^{\log(\hat{h})}$ is a random point chosen by the verifier
- $r_m \in \mathbb{F}^{\log(\hat{m})}$ is a random point chosen by the verifier

Given this claim $y_{\hat{O}}$, the prover now computes the vector $\mathbf{b} \in \mathbb{F}^{\hat{h} \times \hat{s}}$ such that $\mathbf{b}[i] = \beta(\lfloor \frac{i}{\hat{s}} \rfloor, r_h)$. The multi-head matrix multiplication is then proven with the following sum-check:
\begin{equation}
y_{\hat{O}} = \sum_{x \in \{0,1\}^{\log(\hat{h})}} \sum_{y \in \{0,1\}^{\log(\hat{s})}} \mathbf{b}(x, y) \hat{QK}(x, r_s, y)\widetilde{V}(x,y,r_m)
\tag{2}
\end{equation}
The sum-check produces the following claims, for random points $r_x \in \mathbb{F}^{\log(\hat{h})}$, $r_y \in \mathbb{F}^{\log(\hat{s})}$:

- Claim $\mathbf{b}(r_x, r_y)$, which can be efficiency recomputed by the verifier
- Claim $\hat{QK}(r_x, r_s, r_y)$, which can be employed as the claim $y_M$ in the masking proving protocol described [above](#prove-masking)
- Claim $y_{\widetilde{V}} = \widetilde{V}(r_x, r_y, r_m)$, from which the verifier can get the claim $y_{\hat{V}}$ about the input matrix $\hat{V}$ as $y_{\hat{V}} = \hat{V}(r_y, r_x, r_m) = y_{\widetilde{V}}$, because of the property about MLEs of $\widetilde{V}$ and $\hat{V}$ described earlier

### Unpadding
The matrix $\hat{O}$ computed by the previous sum-check has shape $[\hat{s}, \hat{h}*\hat{m}]$, while the actual output matrix of the multi-head attention layer must have shape $[\hat{s}, d]$, where $d = 2^{\lceil \log_2(h*m) \rceil}$. In general $d \le \hat{h}\hat{m}$, and so an *unpadding* operation might be necessary.

Recall that the matrix $\hat{O} \in \mathbb{F}^{\hat{s} \times \hat{h} \times \hat{m}}$ is composed by the column-wise concatenation of $\hat{h}$ heads $\hat{O}_i \in \mathbb{F}^{\hat{s} \times \hat{m}}$. The unpadding operation needs to:

- Get rid of the $\hat{m} - m$ columns in each matrix $\hat{O}_i$
- Discard the last $\hat{h} - h$ padding heads

This operation can be performed by multiplying the matrix $\hat{O}$ with an unpadding matrix $U \in \mathbb{F}^{\hat{h}*\hat{m} \times d}$ computed as:
$$
U[i,j] = \begin{cases}
1 \quad \text{if} \ \lfloor \frac{j}{m} \rfloor < h \wedge \lfloor \frac{i}{\hat{m}} \rfloor = \lfloor \frac{j}{m} \rfloor \wedge i \mod \hat{m} = j \mod m \\
0 \quad \text{otherwise} 
\end{cases}
$$
In other words, the unpadding matrix $U$ has $h*m$ non-zero entries, each corresponding to a *non-garbage* entry in each row of the matrix $\hat{O}$, where we deem an entry as *garbage* if it is computed only from values adding by the initial padding of the multi-head attention layer.

The unpadding operation can be proven with a single sum-check. Given a claim $y_O = O(r_s, r_d)$ for the output matrix $O$, where $r_s \in \mathbb{F}^{\log(\hat{s})}$ and $r_d \in \mathbb{f}^{\log(d)}$ are random points chosen by the verifier, the prover proves the following relationship by sum-check:
$$
y_O = \sum_{x \in \{0,1\}^{\log(\hat{h}\hat{m})}} \hat{O}(r_s, x)U(x,r_d)
$$
The sumcheck produces the followings claims for a random point $r \in \mathbb{F}^{\log(\hat{h}\hat{m})}$:

- Claim $U(r, r_d)$, which can be efficiently recomputed by the verifier
- Claim $\hat{O}(r_s, r)$, which can be employed as claim $y_{\hat{O}}$ in the sum-check protocol described in Equation (2)  