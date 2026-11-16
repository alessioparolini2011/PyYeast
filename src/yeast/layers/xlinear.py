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

from yeast.exceptions import XLayersError, XLayersTypeError, XLayersValueError

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
        if in_features is None:
            raise XLayersValueError("Missing argument: 'in_features' cannot be None!")
        if not isinstance(in_features, int):
            raise XLayersTypeError("Wrong argument: 'in_features' must be an integer")
        self.in_features = in_features
        if out_features is None:
            raise XLayersValueError("Missing argument: 'out_features' cannot be None!")
        if not isinstance(out_features, int):
            raise XLayersTypeError("Wrong argument: 'out_features' must be a integer!")
        self.out_features = out_features
        if not isinstance(exp_rate, int):
            raise XLayersTypeError("Wrong argument: 'exp_rate' must be an integer!")
        if exp_rate <= 0:
            raise XLayersValueError("Wrong value: 'exp_rate' must be higher than zero!")
        self.exp_rate = exp_rate
        self.weight = nn.Parameter(data=(torch.randn(self.out_features, self.in_features)))
        self.bias = nn.Parameter(data=(torch.randn(self.out_features)))

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
            raise XLayersTypeError(f"Wrong argument! 'expansion' must be an integer! You instead set it as a {type(expansion).__name__}")
        if expansion <= 0:
            raise XLayersValueError(f"Wrong argument value! 'expansion' must be higher than 0! You instead set it as {expansion}")