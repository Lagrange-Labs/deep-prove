# Quantisation

In order to prove inference we need to be able to represent the inference trace as finitie field elements. To do this we use the fairly standard practice of *Affine Quantisation*.

## A Brief Example

Say we have a tensor $`T`$ and its internal values are all elements of $`\mathbb{R}`$. To quantise this tensor we first choose the domain we wish to quantise into, lets call this $`[\alpha_{q}, \beta_{q}]`$ with $`\alpha_{q} < \beta_{q}`$ and $`\alpha_{q}, \beta_{q} \in \mathbb{Z}`$. We look at all values present in the tensor $`T`$, write $`\alpha`$ for the minimum value and $`\beta`$ for the maximum. We define the *scaling factor*, $` S\in\mathbb{R}`$ by

$$ S := \frac{\beta - \alpha}{\beta_{q} - \alpha_{q}}. $$

In addition, if $`-\alpha \neq \beta`$ and $` -\alpha_{q} \neq \beta_{q}`$ we define the *zero point*, $`z\in\mathbb{Z}`$ by 

$$ z := \mathrm{Round}\left(\frac{\beta\alpha_{q} -\alpha\beta_{q}}{\beta - \alpha}\right). $$

Then for a real value $`x\in T`$, we link it to the quantised value $` x_{q}\in [\alpha_{q}, \beta_{q}] `$ by the formula 

$$ x = S(x_{q} - z). $$

This can be seen in practice with the following example code:
```rust
{{#include ./../src/tensor.rs}}
# use crate::tensor::Tensor;
let float_tensor = Tensor::<f32>::random(vec![8, 8]); 
let x = 5;
println!("This is x: {}", x);
```