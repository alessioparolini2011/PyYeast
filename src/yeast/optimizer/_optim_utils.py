"""
_optim_utils.py
Authors: Alessio Parolini
This file contains all the resources for calculate and use the Fisher matrix to improve and optimize expansion
All these resources are used from the XOptimWrapper
"""

__all__ = ["_natural_gradient"]

import torch

from yeast.layers import XLinear


def _natural_gradient(layer: XLinear,) -> torch.Tensor:
    """
    This function calculates the natural gradient (using Fisher matrix) for every neuron in the layer,
    adding up the natural gradient of every weight and bias in it.
    Actually, the natural gradient should be the inverse of the number this function calculates, but in this way I save an operation
    :param layer: the layer where you want to calculate natural gradient
    :return: the tensor with the natural grad for every neuron
    """
    w_grad = layer.weight.grad if layer.weight.requires_grad else torch.zeros(size=layer.weight.size())
    w_bias = layer.bias.grad if layer.bias.requires_grad else torch.zeros(size=layer.bias.size())
    return (w_grad ** 2).sum(dim=1) + w_bias ** 2
