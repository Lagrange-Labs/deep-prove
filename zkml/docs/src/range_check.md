# Range Checks
Range checks are performed over a table with a single column of size `1 << BIT_LEN` where `BIT_LEN` is the quantised value size in bits. For a simple example if we wish to constrain that every value in a tensor $` F `$ is in the range $` [0, 2^{3}) `$ then we take the MLE representing $` F `$, which we write as $`F(x)`$ and pass it to the GKR circuit which calculates 

$$\begin{align} \sum_{b\in\mathcal{B}} \frac{-1}{\alpha + F(b)} = \frac{p}{q}. \end{align} $$

The output, $` p `$ and $`q`$ is then used in the final verification performed by the verifier. 