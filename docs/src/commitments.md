# Commitment Schemes

Deep-prove is agnostic to which Polynomial Commitment Scheme to use. However, it uses PCS in two different way during accumulation of claims to only perform a single opening at the end of the protocol.

## Accumulation of different polynomials

As described in the high level overview, we need to commit to all the parameters of the model. Instead of committing to them one by one, we commit to the concatenation of all the polynomials and produce only a single opening proof for which the verifier is able to verify the consistency cheaply. Here is the breakdown:

Let $\textbf{w} = [\textbf{w}_1,...,\textbf{w}_k] \in \mathbb{F}^n$ be a vector and each segment $\textbf{w}_i$ *has size that is power of two*.  Furthermore, we assume that the segments of $\textbf{w}$ *are sorted in decreasing order*. Finally let $f_{\textbf{w}_1},...f_{\textbf{w}_k}$ be the MLEs of the segments and $f_{\textbf{w}}$ the MLE of $\textbf{w}$. 

Given a set of evaluation claim pairs $S = \{(y_1,\textbf{r}_1),...,(y_k,\textbf{r}_k)\}$*, we want to accumulate them in a single pair* $(y,\textbf{w})$ *such that* $y = f_{\textbf{w}}(\textbf{r})$ if and only if all claims are correct. We achieve this using the following protocol:

**1. Verifier**: Given the set $S$ samples $k$ random points $a_1,a_2,...,a_n$ and computes $y_{aggr} = \sum_{i \in [k]}a_it_i$. Finally sends the random points to the prover.

**2. Prover**: Let $\textbf{b}_i \in \mathbb{F}^{|\textbf{w}_i|}$ be the vector such that $\textbf{b}_i[j] = \beta(j,\textbf{r}_i)$ (the $eq()$ polynomial often defined in sumcheck and GKR related papers).

Observe that $y_i = \sum_{j \in [|\textbf{w}_i|]}\textbf{w}_i[j]\textbf{b}_i[j]$. 

Based on this, the prover computes the matrix $\textbf{b} = [a_1\textbf{b}_1,...,a_k\textbf{b}_k]$ *and generates a sumcheck proof for the instance* $y_{aggr} = \sum_{x \in \{0,1\}^{\log n}}f_{\textbf{b}}(x)f_{\textbf{w}}(x)$.


**3. Verifier**: At the end of the sumcheck protocol, the verifier ends up with an evaluation claim $y_{\beta}$ and $y$ of $f_{\textbf{b}}(x)$ and $f_{\textbf{w}}(x)$ at $\textbf{r}$. It can locally verify $y_{\beta}$ in time $k\log n$. Finally outputs the pair $(y,\textbf{r})$.

We use the Fiat Shamir transformation with any PCS to render this protocol non interactive:
* During the **setup** phase, the scheme commits to $f_w$ using the underlying PCS. At the moment, deep-prove is using Basefold as the underlying PCS.
* During the **proving** phase, the prover creates an opening proof for $(y,\textbf{r})$ over $f_w$ using Basefold.

**NOTE**: deep-proves runs a *global* accumulation protocol throughout the proving steps, for all layers, such that it only needs to produce a _single_ opening proof at the end.


## Accumulation for same polynomial

This is a special case of the previous section. Instead of having $k$ different polynomials, we only have one. 
The routine is used to *bind* two evaluation claims produced by two different layers *over the same polynomial*. For example, the input of a dense layer and the output of a RELU layer operate on the same polynomial.  

The major difference with the previous section is that there is no commitment to the polynomial during the setup phase. Specifically, the output is  a randomized evaluation claim as in the previous section. However, the prover simply *delegates* the fact of proving that claim to the **global** accumulation protocol described in the previous section ! 
This protocol is cheaper for the verifier and prover than to use the first protocol even if two polynomials are the same.