"""
_optim_utils.py
Authors: Alessio Parolini
This file contains all the resources for calculate and use the Fisher matrix to improve and optimize expansion
All these resources are used from the XOptimWrapper
"""

__all__ = ["_gss"]

import torch

from yeast.layers import XLinear


def _gss(layer: XLinear,) -> torch.Tensor:
    """
    Gradients Square Sum (gss) function calculates the natural gradient (using Fisher matrix) for every neuron in the layer,
    adding up the natural gradient of every weight and bias in it.
    Actually, the natural gradient should be the inverse of the number this function calculates, but in this way I save an operation
    :param layer: the layer where you want to calculate natural gradient
    :return: the tensor with the natural grad for every neuron
    """
    w_grad = layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(layer.weight)
    w_bias = layer.bias.grad if layer.bias.grad is not None else torch.zeros_like(layer.bias)
    return (w_grad ** 2).sum(dim=1) + w_bias ** 2
