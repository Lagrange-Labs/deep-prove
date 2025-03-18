# Layers
We currently support a number of different layers in our models. We describe them, and the procedure for proving each of them in the sections that follow:

* [Relu](./relu.md), we prove correct execution of this function using a [lookup argument](./lookups.md)
* We need to do a [range Checks](./range_check.md) for requantization.
* [Dense layer](./dense.md) 
* [Convolution](./conv.md)
* [MaxPool](./maxpool.md)