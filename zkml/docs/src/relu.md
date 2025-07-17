# Relu
The size of the Relu table depends on the quantisation bit size. It has two columns, one for inputs and one for outputs, both of which are of size `1 << BIT_LEN`. We give a simple 3 bit example table below:

| Input Column | Output Column |
| :----------: | :-----------: |
|      -4      |       0       |
|      -3      |       0       |
|      -2      |       0       |
|      -1      |       0       |
|       0      |       0       |
|       1      |       1       |
|       2      |       2       |
|       3      |       3       |

At each Relu layer we take the MLE that is the output of the previous layer $` \mathrm{ReluInput}(x) `$ and compute the MLE representing the output after applying Relu element wise, $` \mathrm{ReluOutput}(x) `$. Both of these are then passed to the GKR circuit along with column separation challenges which produces a proof attesting to the fact

$$ \begin{align} \sum_{b\in\mathcal{B}} \frac{-1}{\alpha + \mathrm{ReluInput}(b) + \beta\cdot\mathrm{ReluOutput}(b)} = \frac{p}{q}. \end{align}$$

The values $` p`$ and $`q`$ are then used by the verifier to check that across the entire protocol the combined numerator is zero and the combined denominator is non-zero.
