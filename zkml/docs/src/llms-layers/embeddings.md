# Embeddings

Embeddings is a map where the keys are the token id and the values are the embeddings vector. Informally, this is used to map the user input to weights that that the model can manipulate.

More formally, the embeddings map is $E: \mathbb{T} \rightarrow \{\mathbb{R}\}^{e}$ where $\mathbb{T}$ is 
set of token id and $e$ is the embedding size of the model.

## Inference

The input tensor $x$ is a tensor of tokens IDs of shape $[s,1]$ where $s$ is the sequence length. 
The map is a tensor of all the embeddings of shape $[v,e]$ where $v$ is the vocabulary size of the model. The operation takes every token id $id$ and outputs $E[id]$.
The output shape is therefore $[s,e]$.

## Proving

### Rationale

Proving a non linear map operation is usually implemented using lookup tables. However, that is costly in terms of commitment since it requires to commit to the output. In this case, the output is $[s,e]$ and with gpt2 already, if we consider the maximum length, the output length is of 786432 elements. 

It turns out that we can use a method close to what Lasso/Jolt is doing using one hot encoding vectors of the input. This method only requires 
* proving a matrix multiplication via sumcheck between the table and the one hot encoding to create the correct output
* the verifier to evaluate the one hot encoding at a random point.

So the gist for the verifier boils down to efficiently evaluating the one hot encoding vector at a random point. Given the one hot encoding vector is highly structured, it is efficiently verifiable in $log(v)$ time. 
We choose this approach as even though it increases verifier time, it is strictly better in terms of proving time due to not having to commit to the output.

### Setup

During the setup phase, the table $E$ needs to be committed to; which we call $C_E$.

### Prover

The prover does the following operations:
1. Creates the one hot encoding $H$ of shape $[s,v]$ where $H[i][x[i]] = 1$ and $0$ anywhere else.
    * There are $s$ one hot encoding vectors, concatenated together.
2. Prove $R = H * E$ where $R$ is therefore the correct output of the embeddings layer. To do that, the prover just uses a sumcheck to prove the matrix multiplication as explained in TODO.

At the end, the prover is producing two claims:
1. one claim $c_H$ : $H(r) = y_H$ 
2. one claim $c_E$ : $E(r) = y_E$

$c_E$ is to be proven via PCS opening, using our regular batch claim accumulation proving (TODO LINK).
$c_H$ however is simply given to the verifier. The verifier will evaluate directly the $H$ polynomial efficiently.

### Verifier




