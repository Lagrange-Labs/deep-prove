# Maxpool
Maxpool layers slide a window over each of the channels in the input tensor, returning the maximum value found in each "stride" of the window. If the maxpool window (also called the kernel) has dimension $`k \times k`$ and the input tensor channels have dimension $`n \times n`$ and the stride length of the maxpool kernel is $`s`$ then each of the output channels will have dimension $`\mathrm{ceil}((n - k) / s) \times \mathrm{ceil}((n - k) / s)`$.

## Proving Correct Execution
There are two main components to proving the correct execution of a Maxpool step, a Zerocheck and a [Range Check](./range_check.md).