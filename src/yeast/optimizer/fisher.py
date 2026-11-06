"""
fisher.py
Authors: Alessio Parolini
This file contains all the resources for calculate and use the Fisher matrix to improve and optimize expansion
All these resources are used from the XOptimWrapper
"""

import torch

def _calculate_fisher(weights: torch.Tensor, bias: torch.Tensor)-> torch.Tensor:

    """
    Calculate the Fisher matrix (a special table to know where and how to expand) for a layer
    :param weights: the weights of the layer
    :param bias: the biases of the layer
    :return: the "vertical" tensor with all the matters of each neuron
    """

    if weights.grad is None:

        return torch.zeros(weights.size(0)) # if there's no gradient, return a blank tensor

    if bias.grad is None:

        bias_grad = torch.zeros_like(bias)

    else:

        bias_grad = bias.grad

    matter = [] # the list to contains the matter of each neuron

    for n in range(weights.size(0)):

        w_grad = weights.grad[n]

        b_grad = bias_grad[n]

        fisher = (w_grad ** 2).sum() + (b_grad ** 2)  # every weight/bias is squared

        matter.append(fisher)

    return torch.stack(matter)  # transforming it in a tensor

