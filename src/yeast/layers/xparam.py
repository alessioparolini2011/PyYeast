"""
xparam.py
Authors: Alessio Parolini
This file contains the XParameter class, a subclass of PyTorch nn.Parameter that can grow without changing its identity.
It updates .data in place, so the optimizer keeps tracking the same object during expansion.
"""

__all__ = ["XParameter"]

import torch
from torch import nn, Tensor

from yeast.exceptions import XNetError, XNetTypeError, XNetValueError


class XParameter(nn.Parameter):
    """
    XParameter is a subclass of nn.Parameter that can add new parameters at runtime.
    It keeps the same object identity when it grows, so the optimizer reference stays valid.
    It only stores and grows the data. Splitting and scaling of new parameters is handled by the layer, not here.
    """
    def __init__(self,
                 data: Tensor,
                 requires_grad: bool = True,):
        """
        :param data: the default argument for nn.Parameter
        :param requires_grad: the default argument for nn.Parameter
        """
        super().__init__(data=data, requires_grad=requires_grad)
        self.data = super().data
    def add_params(self,
                   parameters: torch.Tensor,)-> None:
        """
        This method adds new params to the Parameter tensor (.data)
        :param parameters: the new params that will be added to the Parameter tensor
        :return: none
        """
        if not isinstance(parameters, torch.Tensor):
            raise XNetTypeError(f"Wrong argument: 'new_params' must be a PyTorch torch.Tensor instance! You've instead set it as a {type(new_params).__name__}")
        if parameters.size(0) !=  self.data.size(0):
            raise XNetValueError(f"Wrong argument: 'new_params' size isn't compatible with the actual param one!")
        self.data = torch.cat([self.data, parameters], dim=1)
