# LayerNorm

# Description of the Layer

LayerNorm can be thought of as transforming a tensor so that its values follow a normalised standard distribution, where we normalise on a certain dimension. For example if we have an input tensor $` A `$ defined over the reals $` \mathbb{R}`$. Without loss of generality we can assume $`a \in \mathbb{R}^{k\times n}`$ is a matrix then we define

$$ \begin{align*} \mathrm{LayerNorm}(A)_{i,j} := \gamma \cdot \frac{A_{i,j} - \mu_{i}}{\sqrt{\frac{1}{n}\cdot\sum_{l=1}^{n}(A_{i,l} -\mu_{i})^{2} + \epsilon}} + \beta.\end{align*} $$

Here we have that $`\mu_{i}`$ is the mean of the values on the $`i`$-th row, $`\gamma `$ and $`\beta`$ are learned constants and $`\epsilon`$ is a normalisation factor.

## Proving the Layer

To prove the correct execution of LayerNorm we use a combination of lookups and standard sumchecks. The lookup protocol is used to prove correct computation of 

$$ \begin{align*} D_{i} := \frac{1}{\sqrt{\frac{1}{n}\cdot\sum_{l=1}^{n}(A_{i,l} -\mu_{i})^{2} + \epsilon}} \end{align*} $$

and then a standard product sumcheck is used to prove that $`(A_{i,j}-\mu_{i})* D_{i} = \mathrm{LayerNorm}(A)_{i,j}
`$ element-wise.

### Step-by-Step

The prover recieves the input tensor $`A`$ and its corresponding MLE $`A(\bar{x})`$. They use this to compute the input to the lookup table $`\mathrm{LookupIn}`$ and output of the lookup table $`D`$ together with their corresponding MLEs $`\mathrm{LookupIn}(\bar{x})`$ and $`D(\bar{x})`$.

The prover also has the claim about the LayerNorm output which is a point $`\bar{r}`$ and a value $`v`$. They use this to reduce the claim on the output to a claim on $`A(\bar{x})`$ and $`D(\bar{x})`$ by running the sumcheck:

$$ \begin{align} \gamma^{-1}(v-\beta)=\sum_{b\in\mathcal{B}_{n}}(\mathrm{eq}(\bar{r},b) - 2^{k}/n\cdot \mathrm{eq}(2^{-1},\dots,2^{-1},r_{k+1},\dots, r_{n}, b))\cdot A(b)\cdot D(b).\end{align} $$

They run the lookup argument to obtain claims about $`\mathrm{LookupIn}`$ and $`D`$ as well and then run another sumcheck to show that 

$$