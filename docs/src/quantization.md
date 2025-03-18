# Quantization

**NOTE**: This area is a highly active area of r&d for us and may not be up to date with the implementation. 

## Problem

Often ML models represent the weights and inputs as floating point values. Unfortunately, this is a problem for following reasons:

- Floating point arithmetic is slower than integer arithmetic
- Cryptography deals with these models as finite fields

Standard practice is to trade accuracy for efficiency by quantizing weights and inputs. 

## Quantization

Say we know that $r \in \mathbb{R}$ and $r\in [-1, 1]$. To convert $r$ into an B-bit integer, we perform the following affine transformation. 

$$
r = S(q − Z)
$$

$$
q = \text{Round} (\frac{r}{S})  + Z
$$

where

- $r$ is a real-number
- $S \in \mathbb{R}$ is a scaling factor
- $q\in\mathbb{Z}$ is a B-bit representation (e.g., $[-127, 127], [0;255]$, etc.)
- $Z \in \mathbb{Z}$ is the zero-point in the B-bit range

## Model and Input Quantization

**Range**: deep-prove can be parametrized to use different bit length for its quantization. In practice, we've ran our benchmarks with `BIT_LEN=8`, e.g. range is $[-127;128]$.

**Zero Point**: deep-prove assumes a zero centered point. This simplifies formula and proving and is usually assumed by traditional machine learning frameworks.

## References:

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)
- [zkCNN: Zero Knowledge Proofs for Convolutional Neural Network Predictions and Accuracy](https://eprint.iacr.org/2021/673.pdf)