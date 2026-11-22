"""
xparam.py
Authors: Alessio Parolini
This file contains the XParameter class, a subclass of PyTorch nn.Parameter that can grow without changing its identity.
It updates .data in place, so the optimizer keeps tracking the same object during expansion.
"""

__all__ = ["XParameter"]

import torch
from torch import nn, Tensor

from typing import cast

class XParameter(nn.Parameter):
    """
    XParameter is a subclass of nn.Parameter that can add new parameters at runtime.
    It keeps the same object identity when it grows, so the optimizer reference stays valid.
    It only stores and grows the data. Splitting and scaling of new parameters is handled by the layer, not here.
    """
    _shape_before_expansion: torch.Size  # created here because there is no __init__
    def __new__(cls, data: Tensor | None = None, requires_grad: bool = True):
        """
        :param data: the default argument for nn.Parameter
        :param requires_grad: the default argument for nn.Parameter
        """
        instance = cast("XParameter", super().__new__(cls, data=data, requires_grad=requires_grad))
        instance._shape_before_expansion = instance.size()
        return instance

    def add_params(self,
                   parameters: torch.Tensor,
                   dim: int = 0)-> None:
        """
        This method adds new params to the Parameter tensor (.data)
        :param parameters: the new params that will be added to the Parameter tensor
        :param dim: the dimension where the expansion should happen
        :return: none
        """
        if not isinstance(parameters, torch.Tensor):
            raise TypeError(f"Wrong argument: 'new_params' must be a PyTorch torch.Tensor instance! You've instead set it as a {type(parameters).__name__}")
        if not isinstance(dim, int):
            raise TypeError(f"'dim' must be an integer, got {type(dim).__name__}")
        for d in range(self.data.ndim):
            if d != dim and parameters.size(d) != self.data.size(d):
                raise ValueError(f"Wrong argument: 'new_params' size isn't compatible with the actual param one!")
        self._shape_before_expansion = self.size()  # saving the size before expanding
        parameters = parameters.to(device=self.device, dtype=self.dtype)  # synchronizing the new parameter with the original one
        new_data = torch.cat([self.data, parameters], dim=dim)
        with torch.no_grad():
            self.set_(source=new_data)
        if self.grad is not None:
            grad_pad = torch.zeros_like(parameters)
            self.grad = torch.cat([self.grad, grad_pad], dim=dim)

