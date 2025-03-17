# End-to-End Lookup Protocol
Since a model will have many repeated layers to improve efficiency we aim to only produce one multiplicity polynomial per table type. An added bonus to this is that at each step we only have to do proving work proportional to the number of lookups (for lookup steps) or the table size (for mulitpilcity polynomials).

## Splitting Lookups and Tables
As a quick recap the LogUp protocol works as follows. If we have a table $` T:=\{t_{i} \}_{i = 1}^{m} `$ and a set of values $` L:= \{lk_{i}\}_{i=1}^{n} `$ to be looked up then the LogUp protocol proves 

$$\begin{align} \sum_{i = i}^{m}\frac{M_{i}}{\alpha + t_{i}} = \sum_{j=1}^{n}\frac{1}{\alpha + lk_{j}}.  \end{align}$$

Where $`\alpha`$ is a random challenge provided by the verifier and the set $ M $ is defined as

$$\begin{align} M := \{\mathrm{count}(lk_{j})/\mathrm{count}(t_{i}) | t_{i} = lk_{j}\}, \end{align}$$

the number of times the value appears in $` L `$ divided by the number of times it appears in $` T `$. The indexing of $` M `$ lines up with the indexing of $T $, by this we mean $` M_{i} \in M `$ is the number of times $`t_{i} \in T `$ is looked up.

 We use a GKR circuit to prove correct computation of both sides of Equation $`(1)`$  separately. The output of this computation for the right hand side of Equation $`(1)`$ is the claimed numerator and denominator, $` p `$ and $` q `$, such that 

$$ \begin{align}\sum_{j=1}^{n}\frac{-1}{\alpha + lk_{j}} = \frac{p}{q}. \end{align}$$

Like wise the circuits for the left hand side output $` r`$ and $`s`$ such that

$$\begin{align} \sum_{i=1}^{m}\frac{M_{i}}{\alpha + t_{i}} = \frac{r}{s}. \end{align}$$

At the end of the entire protocol we have GKR proofs, together with claimed numerators and denominators for each of the lookups and the tables. The verifier runs the GKR verification procedure for each of the lookup/table proofs and, provided they are all correct, checks that their combined numerator is zero and that their combined denominator is non-zero.

## Witness Generation
After model inference has been completed we can use the inputs and outputs of each of the layers requiring a lookup argument to generate the witness. 

For each of the different table types we employ we collect together all of the lookups into those tables across all steps of the inference. We then use these to calculate a single multiplicity polynomial for each table type. These multiplicity polynomials are then committed and a (batched) PCS opening proof for them is provided to the verifier.

This means that even if we have $`n`$ layers that use $`\mathrm{Table}_{A}`$, we only have to run the GKR circuit for the fractional sumcheck on $`\mathrm{Table}_{A}`$ once. Likewise we only commit and open on multiplicity polynomial for $`\mathrm{Table}_{A}`$.

## Commitments
When proving inference, for every layer that requires a lookup the prover has to provide a commitment and an opening proof to the output of that layer. We also have to provide one commitment for each different table type used.
