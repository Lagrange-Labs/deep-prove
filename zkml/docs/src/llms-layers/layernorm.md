# LayerNorm

# Description of the Layer

LayerNorm can be thought of as transforming a tensor so that its values follow a normalised standard distribution, where we normalise on a certain dimension. For example if we have an input tensor $` A `$ defined over the reals $` \mathbb{R}`$. Without loss of generality we can assume $`a \in \mathbb{R}^{k\times n}`$ is a matrix then we define

$$ \begin{align*} \mathrm{LayerNorm}(A)_{i,j} := \gamma \cdot \frac{A_{i,j} - \mu_{i}}{\sqrt{\frac{1}{n}\cdot\sum_{l=1}^{n}(A_{i,l} -\mu_{i})^{2} + \epsilon}} + \beta.\end{align*} $$

Here we have that $`\mu_{i}`$ is the mean of the values on the $`i`$-th row, $`\gamma `$ and $`\beta`$ are learned constants and $`\epsilon`$ is a normalisation factor.

## Quantised Evaluation

The main difficuly comes from computing the inverse square root term. For this we use a lookup table that takes as input $`\frac{1}{n}\cdot\sum_{l=1}^{n}(A_{i,l} -\mu_{i})^{2}`$ and outputs $`D_{i} = (\frac{1}{n}\cdot\sum_{l=1}^{n}(A_{i,l} -\mu_{i})^{2} + \epsilon)^{-1/2}`$. The final layer output is then calculated by performing the multiplication $`\gamma\cdot (A_{i,j} - \mu_{i})\cdot D_{i} + \beta`$. Then a requantisation layer is inserted afterwards.

## Proving the Layer

To prove the correct execution of LayerNorm we use a combination of lookups and standard sumchecks. The lookup protocol is used to prove correct computation of 

$$ \begin{align*} D_{i} := \frac{1}{\sqrt{\frac{1}{n}\cdot\sum_{l=1}^{n}(A_{i,l} -\mu_{i})^{2} + \epsilon}} \end{align*} $$

and then a standard product sumcheck is used to prove that $`(A_{i,j}-\mu_{i})* D_{i} = \mathrm{LayerNorm}(A)_{i,j}
`$ element-wise.

### Step-by-Step

The prover recieves the input tensor $`A`$ and its corresponding MLE $`A(\bar{x})`$. They use this to compute the input to the lookup table $`\mathrm{LookupIn}`$ and output of the lookup table $`D`$ together with their corresponding MLEs $`\mathrm{LookupIn}(\bar{x})`$ and $`D(\bar{x})`$. Before anything else the prover commits to both $`A(\bar{x})`$ and $`D(\bar{x})`$ and appends the commitments to the transcript.

They run the lookup argument to obtain claims $`\mathrm{LookupIn}(\bar{s}) = u`$ and $`D(\bar{s})=w`$ as well and then run another sumcheck to show that 

$$ \begin{align} \mathrm{LookupIn}(\bar{s}) = \sum_{b\in\mathcal{B}_{n}} \frac{1}{m}\cdot \left(\mathrm{eq}(\bar{s},b) - 2^{k}/m\cdot \mathrm{eq}(2^{-1},\dots,2^{-1},s_{k+1},\dots, s_{n}, b)\right)^{2}\cdot A(b)^{2}.\end{align} $$

The prover also has the claim about the LayerNorm output which is a point $`\bar{r}`$ and a value $`v`$. They use this to reduce the claim on the output to a claim on $`A(\bar{x})`$ and $`D(\bar{x})`$ by running the sumcheck:

$$ \begin{align} \gamma^{-1}(v-\beta)=\sum_{b\in\mathcal{B}_{n}}(\mathrm{eq}(\bar{r},b) - 2^{k}/m\cdot \mathrm{eq}(2^{-1},\dots,2^{-1},r_{k+1},\dots, r_{n}, b))\cdot A(b)\cdot D(b).\end{align} $$

In both cases we are making use of the fact that for any multilinear polynomial $`f(\bar{y})\in\mathbb{F}[X_{1},\dots,X_{l}]`$ we have $`\sum_{b\in\mathcal{B}_{l}} f(b) = 2^{l} \cdot f(2^{-1}, \dots, 2^{-1})`$ to efficiently calculate the row by row average $`\mu_{i}`$.

The prover also performs a sumcheck 

$$ \begin{align} D(\bar{s}) = \sum_{b\in\mathcal{B}_{n}}\mathrm{eq}(\bar{s}, b)\cdot D(b). \end{align} $$

In practice the three Sumchecks above are batched together into a single PIOP leaving the prover with one claim $`A(\bar{t}) = a`$ and $`D(\bar{t}) = d`$. Both of these claims are verified via commitment opening and $`A(\bar{t}) = a`$ is the claim passed to the next layer.

## Verifying 

The verifier receives the LayerNorm proof from the prover. They verify the lookup argument and use the claims output by this to calculate the initial claim to three sumchecks (that are batched together via random challenges).

They run sumcheck verification and use the claimed evaluations $`a`$ and $`d`$ provided by the prover to reconstruct the final sumcheck evaluation (they can compute all $`\mathrm{eq}`$ poly evaluations themselves). Finally they verify the commitment openings for $`A(\bar{t}) = a`$ and $`D(\bar{t}) = d`$ and pass the output claim to the next layer to be verified.