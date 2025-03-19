# Lookup Arguments

There are many types of operation that occur in a Model that don't lend themselves to being proven algebraically. In order to prove correct execution of these operations efficiently we make use of a Lookup Argument. The Lookup argument we use is based on the [GKR variant](https://eprint.iacr.org/2023/1284.pdf) of [LogUp](https://eprint.iacr.org/2022/1530.pdf), but we make some slight adaptations.

The following sections go in to more detail about the protocol as a whole and the tables we use:
* The [End-to-End Lookup Protocol](./end_to_end_lu.md) describes modifications we make to reduce the total amount of proving work.
* [Relu](./relu.md) describes the table used for the Relu activation function.
* [Range Checks](./range_check.md) describes the procedure for performing range checks, which we make use of when requantising and pooling