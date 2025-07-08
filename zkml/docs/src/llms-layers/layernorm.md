# LayerNorm

## Description of the Layer

LayerNorm can be thought of as transforming a tensor so that its values follow a normalised standard distribution, where we normalise on a certain dimension. For example if we have an input tensor $` A `$ defined over the reals $` \mathbb{R}`$. Without loss of generality we can assume $`a \in \mathbb{R}^{k\times n}`$ is a matrix then we define

$$ \begin{align*} \mathrm{LayerNorm}(A)_{i,j} := \gamma \cdot \frac{A_{i,j} - \mu_{i}}{\sqrt{\frac{1}{n}\cdot\sum_{l=1}^{n}(A_{i,l} -\mu_{i})^{2} + \epsilon}} + \beta.\end{align*} $$

Here we have that $`\mu_{i}`$ is the mean of the values on the $`i`$-th row, $`\gamma `$ and $`\beta`$ are learned constants and $`\epsilon`$ is a normalisation factor.

## Proving the Layer

To prove the correct execution of LayerNorm we use a combination of lookups and standard sumchecks. The lookup protocol is used to prove correct computation of 

$$ \begin{align*} D_{i} := \frac{1}{\sqrt{\frac{1}{n}\cdot\sum_{l=1}^{n}(A_{i,l} -\mu_{i})^{2} + \epsilon}} \end{align*} $$

and then a standard product sumcheck is used to prove that $`(A_{i,j}-\mu_{i})* D_{i} = \mathrm{LayerNorm}(A)_{i,j}
`$ element-wise.