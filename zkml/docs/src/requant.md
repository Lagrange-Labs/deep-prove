# Requantisation

After performing operations like convolution or matrix multiplication on quantised values the outputs will no longer have the same bit size. In order to make efficient use of lookups we want to keep the table sizes small and the table size is directly linked to the input bit size. So after any operation involving multiplication and addition, we requantise.

## Simulating Floating Point Multiplication

We use symmetric quantisation throughout (so no shift for zero points), for more infomation refer to the [quantisation](./quantisation.md) chapter. For sake of example say we have the simple multiplication of floating point numbers 

$$ a = b\cdot c $$

and these values are quantised using scaling factors $`S_{a}, S_{b}`$ and $`S_{c}`$ respectively leading to the equality

$$ S_{a}\cdot a_{q} = S_{b}\cdot b_{q} \cdot S_{c}\cdot c_{q}. $$

Then in the quantised world, if $`a_{q}, b_{q}`$ and $`c_{q}`$ are 8-bit integers we have 

$$ a_{q} = \frac{S_{b}\cdot S_{c}}{S_{a}} \cdot b_{q} \cdot c_{q}, $$

and writing $` M = S_{b}\cdot S_{c} / S_{a}`$ we have $` 0< M \leq 1`$. We can express $`M `$ in normalised form as $`2^{-k}\cdot \epsilon`$ for some $`k\in \mathbb{N}`$ and $`\epsilon`$ satisfies $`0.5<\epsilon \leq 1`$. Importantly this means that if we are using the `f32` data type, then $`\epsilon`$ has 24 fractional bits, hence we can express it as a fixed point multiplier with scale factor $`2^{25}`$.

Thus to requantise we take $`b_{q} \cdot c_{q}`$, we multiply by $`\epsilon\cdot 2^{25}`$, add $`2^{25 + k - 1}`$ to account for rounding and then right shift by $`25 + k`$. Putting this altogether yields

$$ a_{q} = (b_{q}\cdot c_{q} \cdot\epsilon\cdot 2^{25} + 2^{25 + k - 1}) >> (25 + k). $$

## Proving Via Bit-Wise Decomposition

Inputs to the requantisation operation will have a maximum bit size, which we denote here by $` t`$, thus after multiplication by $`\epsilon \cdot 2^{25}`$ they are at most $`t + 25 `$ bit signed integers. If we write $`\mathrm{Input}(\bar{x})`$ for the MLE representing the input tensor, then after decomposition we have

$$ \begin{align} \epsilon\cdot 2^{25}\cdot\mathrm{Input}(\bar{x}) + 2^{25 + k - 1} = \sum_{i=0}^{t + 23}2^{i}\cdot\mathrm{Bit}_{i}(\bar{x}) - 2^{t + 24}\cdot\mathrm{Bit}_{t + 24}(\bar{x}). \end{align}$$

Now we know that the bottom $`25 + k`$ bits of each value will be shifted away, this means the output MLE, denoted $`\mathrm{Output}(\bar{x})`$, depends only on the polynomials $`\{\mathrm{Bit}_{j}(\bar{x})\}_{j=25 + k}^{t+24}`$. 

Write $`\mathrm{Max}`$ for the absoloute value of the largest integer in the quantisation range. Then we must clamp if the absoloute value of the recombination of the polynomials in the set above is grater than $`\mathrm{Max}`$. To prove this we supply additional polynomials as "advice". The values in the advice are constructed as 

$$ \mathrm{Advice}(\bar{x}) = \begin{cases} 2^{t - k}+\mathrm{Max} -v(\bar{x}) & \text{if } \ \mathrm{Bit}_{t+24}(\bar{x}) = 0 \\
2^{t - k}-\mathrm{Max} -v(\bar{x}) & \text{if } \ \mathrm{Bit}_{t+24}(\bar{x}) = 1
\end{cases} $$

In the above $`v(\bar{x})=\sum_{j=25 +k}^{t-23}2^{j-25-k}\cdot\mathrm{Bit}_{j}(\bar{x}) - 2^{t -1 -k}\cdot\mathrm{Bit}_{t+24}(\bar{x})`$.

 Importantly, the evaluations of $`\mathrm{Advice}(\bar{x})`$ are values whose most significant bit is $`1`$ if no clamping is required and $`0`$ otherwise. So we decompose $`\mathrm{Advice}(\bar{x})`$ into bit-wise polynomials as well, we denote the most significant bit polynomial by $`\mathrm{Clamp}(\bar{x})`$. Then we have

 $$ \begin{align*} \mathrm{Output}(\bar{x}) = \sum_{\bar{b}\in\{0,1\}^{n}}&\mathrm{eq}(\bar{x}, \bar{b})\cdot (\mathrm{Clamp}(\bar{b})\cdot v(\bar{b}) +\\ 
 & \mathrm{Max}\cdot(1 - \mathrm{Clamp}(\bar{b}))\cdot (1 - 2\cdot \mathrm{Bit}_{t+24}(\bar{b})) ). \end{align*} $$

 This is constrained via a sumcheck on all the individual bit polynomials. The same sumcheck also constrains that all the claimed bit polynomials are boolean valued. After this sumcheck we have claimed values for the bit polynomials, these are passed to a second sumcheck that checks the correct construction of $`\mathrm{Advice}(\bar{x})`$ and also enforces the booleanity check on any bit polynomials that weren't included in the first sumcheck (i.e. the ones corresponding to bits that get shifted away).

 The claims output by this second sumcheck correspond precisely to the polynomials on the right hand side of Equation (1). So the verifier performs the recombination described in Equation (1), subtracts the rounding constant, multiplies by the field inverse of $`2^{25}\cdot\epsilon`$ and uses this as the claim about $`\mathrm{Input}(\bar{x})`$.
