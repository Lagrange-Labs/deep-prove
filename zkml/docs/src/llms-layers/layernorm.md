# LayerNorm

# Description of the Layer

LayerNorm can be thought of as transforming a tensor so that its values follow a normalised standard distribution, where we normalise on a certain dimension. For example if we have an input tensor $` A `$ defined over the reals $` \mathbb{R}`$. Without loss of generality we can assume $`a \in \mathbb{R}^{k\times n}`$ is a matrix then we define

$$ \begin{align*} \mathrm{LayerNorm}(A)_{i,j} := \gamma \cdot \frac{A_{i,j} - \mu_{i}}{\sqrt{\frac{1}{n}\cdot\sum_{l=1}^{n}(A_{i,l} -\mu_{i})^{2} + \epsilon}} + \beta.\end{align*} $$

Here we have that $`\mu_{i}`$ is the mean of the values on the $`i`$-th row, $`\gamma `$ and $`\beta`$ are learned constants and $`\epsilon`$ is a normalisation factor.

## Different Formula for Standrad Deviation

In the above equation for $`\mathrm{LayerNorm}`$ we used the formula $`\sigma = \sqrt{\frac{1}{n}\cdot \sum_{l=1}^{n}(A_{i,l} - \mu_{i})^{2} + \epsilon}`$ for the standard deviation (plus the $`\epsilon`$ normalisation factor). We would like to avoid multiplying by $`1/n`$ where possible because when dealing with quantised integers this introduces some error. To avoid this we use a different formula derived below, we write $`\hat{\mu}_{i} := n\cdot \mu_{i}`$ and $`\hat{\nu_{i}}:= \sum_{l=1}^{n} A_{i,l}^{2}`$.

$$ \begin{align*} \sigma^{2} &= \frac{1}{n}\cdot \sum_{l=1}^{n}(A_{i,l} - \mu_{i})^{2} + \epsilon \\ &= \frac{1}{n}\cdot \sum_{l=1}^{n}(A_{i,l}^{2}-2\cdot A_{i,l}\cdot\mu_{i} + \mu_{i}^{2}) + \epsilon \\ &= \frac{1}{n}\sum_{l=1}^{n}\left(A_{i,l}^{2} -2\cdot A_{i,l}\cdot \frac{\sum_{k=1}^{n} A_{i,k}}{n} + \frac{(\sum_{k=1}^{n} A_{i,k})^{2}}{n^{2}}\right) + \epsilon \\ &= \sum_{l=1}^{n}\frac{A_{i,l}^{2}}{n} - 2\cdot \frac{\sum_{k=1}^{n} A_{i,k}}{n}\cdot \frac{\sum_{l=1}^{n} A_{i,l}}{n}+ \frac{(\sum_{k=1}^{n} A_{i,k})^{2}}{n^{2}} + \epsilon \\ &= \frac{1}{n^{2}}\cdot (n\cdot \hat{\nu}_{i} - 2\cdot \hat{\mu}_{i}^{2} + \hat{\mu}_{i}^{2} + n^{2}\cdot\epsilon) \\ \implies \sigma &= \frac{\sqrt{n\cdot \hat{\nu}_{i} - \hat{\mu}_{i}^{2} + n^{2}\cdot\epsilon}}{n}.\end{align*} $$

This allows us to pass $`n\cdot \hat{\nu}_{i} - \hat{\mu}_{i}^{2}`$ as input to the lookup table and it returns $`\frac{1}{\sqrt{n\cdot \hat{\nu}_{i} - \hat{\mu}_{i}^{2} + n^{2}\cdot\epsilon}}`$.

## Precision

When we calculate the term $`n\cdot \hat{\nu}_{i} - \hat{\mu}_{i}^{2}`$ to pass to the lookup we don't actually use the full value as the range of possible values it could take is far too large. Instead we only use the most significant $`2b_{Q}`$ bits, where $`b_{Q}`$ is the quantisation bit-length.

To see why we do this consider the following example. We pick two numbers $`3.4372`$ and $`4.5133`$ and their product is $`15.51311476`$. If we round both to three significant figures then we have $`3.44`$ and $`4.51`$, their product is $`15.5144`$ which we can see differs at the 5th significant figure, so using anything beyond this isn't going to give any more accuracy in the final result (as it is already wrong).

## Quantised Evaluation

The main difficuly comes from computing the inverse square root term. For this we use a lookup table that takes as input $`\sum_{l=1}^{n}n\cdot A_{i,l}^{2} -\mu_{i}^{2}`$ and outputs $`D_{i} = (\sum_{l=1}^{n}n\cdot A_{i,l}^{2} -\mu_{i}^{2} + \epsilon)^{-1/2}`$. The final layer output is then calculated by performing the multiplication $`\gamma\cdot (n\cdot A_{i,j} - \mu_{i})\cdot D_{i} + \beta`$. 

This output involves a degree 4 multiplication and so the bit size of the output is large, too large to apply our normal requantisation technique of applying $`m = s_{\mathrm{in}}/s_{\mathrm{out}}`$ via its normal form $`m = 2^{-t}\cdot \epsilon`$ using a fixed pointsclae factor. To still allow us to perform a requantisation we set $`s_{\mathrm{out}}`$ to be such that the requantisation requires only a right shift (i.e. we pick $`s_{\mathrm{out}}`$ such that $`\epsilon = 1`$).

## Proving the Layer

To prove the correct execution of LayerNorm we use a combination of lookups and standard sumchecks. The lookup protocol is used to prove correct computation of 

$$ \begin{align*} D_{i} := \frac{1}{\sqrt{\sum_{l=1}^{n}n\cdot A_{i,l}^{2} -\mu_{i}^{2} + \epsilon}} \end{align*} $$

and then a standard product sumcheck is used to prove that $`\gamma \cdot (n\cdot A_{i,j}-\mu_{i})* D_{i} + \beta = \mathrm{LayerNorm}(A)_{i,j}
`$ element-wise.

Throughout we use the fact that if $`\hat{\mu}_{i} = \sum_{l=1}^{n} A_{i,l}`$ then the polynomial that represents the tensor with each value on row $`i`$ being $`\hat{\mu}_{i}`$ (so if the original row is $`A_{i, 1},\dots,A_{i,n}`$ this new tensor has row $`\hat{\mu}_{i},\dots,\hat{\mu}_{i}`$) can be computed by

$$ \begin{align*} \hat{\mu}(r_{1},\dots, r_{k}) = \sum_{b\in\mathcal{B}_{k}}2^{\lceil\log{n}\rceil}\cdot\mathrm{eq}(2^{-1},\dots,2^{-1},r_{\lceil\log{n}\rceil},\dots, r_{k}, b)\cdot A(b) \end{align*} $$

### Step-by-Step

The prover recieves the input tensor $`A`$ and its corresponding MLE $`A(\bar{x})`$. They use this to compute the input to the lookup table $`\mathrm{LookupIn}`$ and output of the lookup table $`D`$ together with their corresponding MLEs $`\mathrm{LookupIn}(\bar{x})`$ and $`D(\bar{x})`$. 

To compute $`\mathrm{LookupIn}`$ the prover calculates $`\sum_{l=1}^{n}n\cdot A_{i,l}^{2} -\mu_{i}^{2}`$. The outputs of this sum have large bit size, too large for a single lookup table, so if the quantisation bit length is $`b_{q}`$ we use the fact that only the most significant $`2b_{q}`$ bits of the sum have any real precision (any bits less significant than this are calculated via terms that involve rounding error). We split the most significant $`2b_{q}`$ (see [this](./layernorm.md#precision) for more info) bits of the sum off to become $`\mathrm{LookupIn}`$ and range check the remainder.

 Now the prover commits to both $`\mathrm{LookupIn}(\bar{x})`$, $`D(\bar{x})`$ and the range check lookup, appending the commitments to the transcript.

They run the lookup argument to obtain claims $`\mathrm{LookupIn}(\bar{s_{1}'}) = u`$, $`D(\bar{s_{1}'})=w`$ and $`\mathrm{Range}(\bar{s_{2}'}) = t`$. They then run another sumcheck to force all of these polynomials to be evaluated on the same point $`s`$. We note that the point $`s`$ has $`\lceil\log{n}\rceil`$ fewer variables than the polynomial $`A(\bar{x})`$ does.

Now that everything is evaluated on the same point they check via sumcheck that

$$ \begin{align} 2^{\mathrm{rangebits}}\cdot\mathrm{LookupIn}(\bar{s}) + \mathrm{Range}(\bar{s}) =& \sum_{b\in\mathcal{B}_{m}} \mathrm{eq}(2^{-1},\dots,2^{-1},s, b)(n\cdot 2^{\lceil\log{n}\rceil} A(b)^{2} -\mu(b)^{2})\end{align} $$

where each "row" of $`\mu(\bar{x})`$ is the sum $`\sum_{l=1}^{n} A_{i,l}`$ repeated $`2^{\lceil\log{n}\rceil}`$ times.

The prover also has the claim about the LayerNorm output which is a point $`\bar{r}`$ and a value $`v`$. They use this to reduce the claim on the output to a claim on $`A(\bar{x})`$ and $`D(\bar{x})`$ by running the sumcheck:

$$ \begin{align} v=\sum_{b\in\mathcal{B}_{m}}\mathrm{eq}(\bar{r},b)\cdot (\gamma(b)\cdot \hat{D}(b)\cdot(n\cdot A(b) -\mu(b)) + \beta(b))\end{align} $$

Here $`\hat{D}(\bar{x})`$ is an extension of $`D(\bar{y})`$ such that $`D(\bar{y}) = \hat{D}(a_{1},\cdots,a_{\lceil\log{n}\rceil,\bar{r}})`$ for any choice of the $`a_{i}`$.

These two checks are batched together, the claim the prover creates on $`\hat{D}`$ is verified by commitment opening and then we use the claims on $`A`$ and $`\mu`$ to show that $`\mu(r) = \sum_{b\in\mathcal{B}_{m}} 2^{\lceil\log{n}\rceil}\cdot \mathrm{eq}(2^{-1},\dots,2^{-1},r_{\lceil\log{n}\rceil},\dots,r_{m}, b)\cdot A(b)`$.

This final sumcheck provides use with the claim that is passed to the next layer.

