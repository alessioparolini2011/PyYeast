"""
xlinear.py
Authors: Alessio Parolini
This file contains the class XLinear, built for provide the user a way to build and manage a SELL (Self Expanding Linear Layer).
Using a low-level interaction with original resources of PyTorch, this class is full-compatible with it and uses its optimization and advantage.
"""

__all__ = ["XLinear"]

import torch
from torch import nn
from torch.nn import functional as f

from yeast.nn.xparam import XParameter

from yeast.exceptions import YeastException, YeastValueError, YeastTypeError, YeastRuntimeError


class XLinear(nn.Module):
    """
    This class provide the structures and the functions to build a linear layer that can expand itself.
    It's a subclass of nn.Module, so it's PyTorch based and uses its optimized calculation system and
    its other resources like autograd or forward
    """
    def __init__(self,
                 out_features: int,
                 in_features:int,
                 exp_rate: int = 1,
                 ):
        """
        :param out_features: number of output features. Must be an integer higher than 0.
        :param in_features: number of input features. Must be an integer higher than 0. Required, it is not inferred.
        :param exp_rate: how much the extension is increased. Must be an integer higher than 0. Default is 1.
        """
        super().__init__()
        if not isinstance(in_features, int):
            raise TypeError(f"Wrong argument! 'in_features' must be an integer! You've instead set it as a {type(in_features).__name__}")
        if in_features <= 0:
            raise ValueError(f"Wrong argument value! 'in_features' must be higher than 0! You've instead set it as {in_features}")
        self.in_features = in_features
        if not isinstance(out_features, int):
            raise TypeError(f"Wrong argument! 'out_features' must be an integer! You've instead set it as a {type(out_features).__name__}")
        if out_features <= 0:
            raise ValueError(f"Wrong argument value! 'out_features' must be higher than 0! You've instead set it as {out_features}")
        self.out_features = out_features
        if not isinstance(exp_rate, int):
            raise TypeError(f"Wrong argument! 'exp_rate' must be an integer! You've instead set it as a {type(exp_rate).__name__}")
        if exp_rate <= 0:
            raise ValueError(f"Wrong argument value! 'exp_rate' must be higher than 0! You've instead set it as {exp_rate}")
        self.exp_rate = exp_rate
        self.weight = XParameter(data=(torch.randn(self.out_features, self.in_features)))
        self.bias = XParameter(data=(torch.randn(self.out_features)))  # using the special variant of nn.Parameter for the parameters
        self._last_expansion = 0

    def expand(self, expansion:int,
               weight_template: torch.Tensor | None,
               bias_template: torch.Tensor | None,)-> None:
        """
        This function expands the params of the layer, and it can accept template weight and bias to create new params starting from the data of another ones
        :param expansion: how many neurons it creates
        :param weight_template: the weight template
        :param bias_template: the bias template
        :return: none
        """
        if not isinstance(expansion, int):
            raise TypeError(f"'expansion' must be an integer, got {type(expansion).__name__}")
        if expansion <= 0:
            raise ValueError(f"'expansion' must be >0, got {expansion}")
        if weight_template is not None:
            if not isinstance(weight_template, torch.Tensor):
                raise TypeError(f"'weight_template' must be a torch.Tensor instance, got {type(weight_template).__name__}")
            if weight_template.size(1) != self.in_features:
                raise ValueError(f"'weight_template' must have {self.in_features} columns, got {weight_template.size(1)}")
        if bias_template is not None:
            if not isinstance(bias_template, torch.Tensor):
                raise TypeError(f"'bias_template' must be a torch.Tensor instance, got {type(bias_template).__name__}")
            if bias_template.numel() != 1:
                raise ValueError(f"'bias_template' must have 1 element, got {bias_template.numel()}")
        expansion *= int(self.exp_rate)
        if weight_template is None:  # it seems wrong, but in this way arguments check and operations are separated
            new_weight = torch.randn(expansion, self.in_features, device=self.weight.device, dtype=self.weight.dtype) * 1e-2
        else:
            new_weight = weight_template.repeat(expansion - 1, 1).to(device=self.weight.device, dtype=self.weight.dtype)  # creating new weight-1: one is the same as the template
            noise = torch.randn_like(new_weight) * 1e-2
            new_weight = torch.cat([new_weight + noise, weight_template], dim=0)
        if bias_template is None:
            new_bias = torch.randn(expansion, device=self.bias.device, dtype=self.weight.dtype) * 1e-2
        else:
            new_bias = bias_template.repeat(expansion - 1).to(device=self.bias.device, dtype=self.bias.dtype)  # creating new bias, leaving a space for the identical one (like with weight)
            noise = torch.randn_like(new_bias) * 1e-2
            new_bias = torch.cat([new_bias+noise, bias_template], dim=0)
        self.weight.add_params(new_weight, dim=0)
        self.bias.add_params(new_bias, dim=0)
        self._last_expansion += expansion

    def _input_adapting(self, x) -> None:
        """
        This function is essential for the correct working of the expansion.
        In fact, when a previous layer expands itself,
        :param x: the input tensor
        :return: None
        """
        x_size = x.size(1)  # get the number of inputs in the input tensor
        extra_size = x_size - self.in_features  # check if are more than the previous epoch
        if extra_size == 0:  # if it's not it returns
            return
        self.weight.add_params(parameters=torch.zeros(self.weight.size(0), extra_size), dim=1)
        self.in_features = self.weight.size(1)  # update the in_features value

    def forward(self, x):
        """
        Get the input and process it using the layer parameters.
        It auto-adapts itself if the X isn't the same of the precedent epoch (for example: a previous layer expanded itself)
        :param x: the input tensor
        :return: the layer output
        """
        self._input_adapting(x)  # adapting to eventual expansion
        x = f.linear(x, self.weight, self.bias)  # computing the data
        return x
