# Padding
Since operations like pooling and convolutions can reduce the size of the input tensor, sometimes we wish to pad the inputs before hand so that the size remains unchanged. At the moment we only support 2D-padding so we only cover this case here.

## Padding as a Linear Operator
Let us write $T$ for the tensor that we pass as input to the FFT-convolution and write $S$ for the corresponding unpadded version that is input to the layer. 

 Assume that $S$ has dimensions $n\times h\times w$ then $T$ has dimensions $n \times (a + h + b) \times (c + w + d)$. Due to how the convolution works we can view both $T$ and $S$ as $n$ lots of $(a + h + b) \times (c + w + d)$ and $h \times w$ matrices respectively, we label each of these matrices $T_{i}$ and $S_{i}$. More formally this translates to tensor products

$$
\begin{align*}
T &= \bigotimes_{i=1}^{n} T_{i} \\
S &= \bigotimes_{i=1}^{n} S_{i}.
\end{align*}
$$

We can define the padding function for one of the $S_{i}\in \mathbb{F}^{h\times w}$ by the linear map

$$
\begin{align*}
\mathrm{Pad}(a,b,c,d): \mathbb{F}^{h \times w} \rightarrow \mathbb{F}^{(a + h + b) \times (c + w + d)}
\end{align*}
$$

it is defined by 

$$
\begin{align*}
\mathrm{Pad}(a, b, c, d)(A) &:= L\cdot A \cdot R,
\end{align*}
$$

where $L\in \mathbb{F}^{(a + h + b)\times h}$ and $R\in \mathbb{F}^{w\times (c + w +d)}$ are the matrices defined by 

$$
\begin{align*}
L &:= \begin{pmatrix}
0_{a\times h} \\
I_{h} \\
0_{b\times h}
\end{pmatrix} \\
R &:= \begin{pmatrix}
0_{w\times c} \\
I_{w} \\
0_{w\times d}
\end{pmatrix},
\end{align*}
$$

with $I_{k}$ the $k\times k$ identity matrix and $0_{q\times r}$ the $q\times r$ matrix where every element is $0$. Importantly we note that $L$ is $h$-sparse and $R$ is $w$-sparse. 

By standard properties of tensor products and linear maps this means we can define the function 

$$
\begin{align*}
\mathrm{TensorPad}(a,b,c,d): \mathbb{F}^{n \times h \times w} \rightarrow \mathbb{F}^{n \times (a + h + b) \times (c + w + d)},
\end{align*}
$$

which is the unique linear map satisfying

$$
\begin{align*}
\mathrm{TensorPad}(a,b,c,d)(S)=\bigotimes_{i=1}^{n}\mathrm{Pad}(a,b,c,d)(S_{i}).
\end{align*}
$$

## Proving

WLOG assume that $n=2^{l}$ for some $l$, then (after possibly applying some entirely different padding), we can write $T(X)$ and $S(Y)$ for the multilinear extensions of $T$ and $S$ respectively. Similarly each of the $T_{i}$ and $S_{i}$ matrices from before correspond to fixing the high $l$ variables of $T(X)$ and $S(Y)$ to the point $C_{i}=(C_{i,1}\dots C_{i,l}) \in \{0,1\}^{l}$ such that $i =\sum_{j=1}^{l} 2^{j-1}C_{i,j}$. Then we have 

$$
\begin{align*}
T(X) = \sum_{i=1}^{n}\chi_{C_{i}}(X_{\mathrm{hi}})\cdot T_{i}(X_{\mathrm{low}}),
\end{align*}
$$

Similarly for $S(Y)$ we have 

$$
\begin{align*}
S(Y) = \sum_{i=1}^{n}\chi_{C_{i}}(Y_{\mathrm{hi}})\cdot S_{i}(Y_{\mathrm{low}}).
\end{align*}
$$

We can also express the $L$ and $R$ matrices from the $\mathrm{Pad}(a,b,c,d)$ function as multilinear extensions $L(W)$ and $R(Z)$. This lets us establish a relation between the $T_{i}(X_{\mathrm{low}})$ and the $S_{i}(Y_{\mathrm{low}})$ via 

$$
\begin{align*}
T_{i}(X_{\mathrm{low},1},X_{\mathrm{low},2}) = \sum_{(Y_{\mathrm{low},1},Y_{\mathrm{low},2})}L(X_{\mathrm{low},1},Y_{\mathrm{low},1})\cdot S_{i}(Y_{\mathrm{low},1},Y_{\mathrm{low},2})\cdot R(Y_{\mathrm{low},2},X_{\mathrm{low},2}).
\end{align*}
$$

By randomly sampling some variables we can prove the above equation with a sumcheck, namely a sparse-dense sumcheck.

## Batching all the sparse-dense sumchecks together

If we sample $(r_{1},\dots,r_{l})\in\mathbb{F}$ we observe the following   

$$
\begin{align*}
T(X_{\mathrm{low}},r_{1},\dots,r_{l})&= \sum_{i=1}^{n}\chi_{C_{i}}(r_{1},\dots,r_{l})\cdot T_{i}(X_{\mathrm{low}}) \\
&= \sum_{i=1}^{n}\chi_{C_{i}}(r_{1},\dots,r_{l})\cdot \sum_{(Y_{\mathrm{low},1}, Y_{\mathrm{low},2})}L(X_{\mathrm{low},1}, Y_{\mathrm{low},1})\cdot S_{i}(Y_{\mathrm{low},1},Y_{\mathrm{low},2})\cdot R(Y_{\mathrm{low},2}, X_{\mathrm{low},2}) \\
&= \sum_{(Y_{\mathrm{low},1}, Y_{\mathrm{low},2})}L(X_{\mathrm{low},1}, Y_{\mathrm{low},1})\cdot\left(\sum_{i=1}^{n}\chi_{C_{i}}(r_{1},\dots,r_{l})\cdot  S_{i}(Y_{\mathrm{low},1},Y_{\mathrm{low},2})\right)\cdot R(Y_{\mathrm{low},2}, X_{\mathrm{low},2}) \\
&= \sum_{(Y_{\mathrm{low},1}, Y_{\mathrm{low},2})}L(X_{\mathrm{low},1}, Y_{\mathrm{low},1})\cdot S(Y_{\mathrm{low},1},Y_{\mathrm{low},2}, r_{1}, \dots, r_{l})\cdot R(Y_{\mathrm{low},2}, X_{\mathrm{low},2})
\end{align*}
$$

the last equality means that we can perform a single [split sumcheck](./split_sumcheck.md) for the whole tensor by fixing variables.

## Verifying

Since the $L$ and $R$ matrices are incredibly sparse and fixed by the model, the verifier can directly evaluate the claims output by the [split sumcheck](./split_sumcheck.md).