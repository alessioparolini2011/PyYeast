"""
xoptim.py
Authors: Alessio Parolini
In this file you can find resources I used to adapt torch.optim.Optimizer to PyYeast componentes
"""

__all__ = ["XOptimizer"]

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch import nn

from yeast.exceptions import YeastRuntimeError
from yeast.nn.xlinear import XLinear
from yeast.nn.xparam import XParameter
from ._optim_utils import _gss

from typing import Iterable, Any
from collections import defaultdict


class XOptimizer:
    """
    This class is an improved version of the previous adapter of the PyTorch optimizers, XOptimWrapper.
    The different is that now you don't create the optimizer outside, and then you use a wrapper,
    instead you create the PyTorch optimizer and the SENN-compatible version with the same object
    """

    def __init__(self, algorithm: type[Optimizer],
                 model: nn.Module,
                 params: Iterable[Tensor],
                 max_params: int,
                 wait_for_expansion: int = 5,
                 **kwargs: Any, ):
        """
        :param algorithm: the optimizer algorithm you choose for the optimizer. Use the same name you find on PyTorch (Adam, SGD...)
        :param model: the PyTorch nn.Module
        :param params: the argument params is the same you'd pass to a classic optimizer
        :param max_params: max total parameters allowed, same limit as XOptimWrapper
        :param wait_for_expansion: how many times it skips the expansion after it has done one
        :param kwargs: the others arguments for the optimizer you choose
        """
        if not issubclass(algorithm, Optimizer):
            raise TypeError("Wrong argument: 'algorithm' must be a subclass of PyTorch 'Optimizer'!")
        self.optimizer = algorithm(params=params, **kwargs)  # creates the classic optimizer
        if not isinstance(model, nn.Module):
            raise TypeError(f"'model' must be a PyTorch nn.Module instance, got {type(model).__name__}")
        self.model = model
        if not isinstance(max_params, int):
            raise TypeError(f"'max_params' must be an integer, got {type(max_params).__name__}")
        if max_params < self.model_size():
            raise ValueError(f"'max_params' cannot be lower than the actual size of the given module: max_params = {max_params} and model size = {self.model_size()}")
        self.max_params = max_params
        if not isinstance(wait_for_expansion, int):
            raise TypeError(f"'wait_for_expansion' must be an integer, got {type(wait_for_expansion).__name__}")
        if wait_for_expansion < 0:
            raise ValueError(f"'wait_for_expansion' must be >= 0, got {wait_for_expansion}")
        self.wait_for_expansion = wait_for_expansion
        self._cooldown = 0
        self.xlayers_list: list[XLinear] = self._get_xlayers()  # this list is for containing the references to the expanding nn

    """
    These are some resources to create a safe bridge to the PyTorch optimizer I've created inside XOptimizer.
    It follows other libraries examples and standards.
    """

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        self.optimizer.param_groups = value

    @property
    def state(self) -> defaultdict[Tensor, Any]:
        return self.optimizer.state

    @state.setter
    def state(self, value: defaultdict[Tensor, Any]) -> None:
        self.optimizer.state = value

    def step(self, closure: Any = None) -> Any:
        # insert custom pre-step logic here
        loss = self.optimizer.step(closure=closure)
        # insert custom post-step logic here
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, item: str) -> Any:
        if item == "optimizer":
            raise AttributeError()
        return getattr(self.optimizer, item)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.optimizer)})"


    def xstep(self, expand = True) -> None:
        """
        This function is designed to expand the nn, using optimized algorithms,
        also updating their dicts (like 'state_dict' or 'param_groups') so the PyTorch optimizer doesn't break
        Also calls the classic .step() method from the original optimizer
        :param expand: if it's False only synchronize params and do the classic .step()
        :return: None
        """
        for layer in self.xlayers_list:
            self._sync_params(layer.weight)
            self._sync_params(layer.bias) # this operation is SUPER IMPORTANT to synchronize the parameters added during the forward (with _input_adapting) due to a previous layer expansion
        self._cooldown += 1
        if expand and self._cooldown >= self.wait_for_expansion:
            self._cooldown = 0
            for layer in self.xlayers_list:
                bias = layer.bias
                weight = layer.weight  # saves the two old parameters (they're necessary for get the position in optimizer dicts)
                gss = _gss(layer=layer)  # get the tensor with the sum of gradients square
                max_idx = torch.argmax(gss).item()  # get the index of the highest sum
                max_gss = torch.max(gss)  # gets the value of the highest gss
                w_template = layer.weight[max_idx : max_idx + 1, :]  # using slicing to keep 2D
                b_template = layer.bias[max_idx: max_idx + 1]  # getting the weight and bias (I used slicing to keep 1D) template (the ones with the highest gradients square sum)
                std, mean = torch.std_mean(gss, correction=0)
                z_score = round(((max_gss - mean) / (
                        std + 1e-8)).item())  # calculating the z-score, it knows how much the highest value deviates from the mean. It's rounded, so it can be used for the expansion
                if layer.exp_rate > 0 and z_score >= 2:  # the first condition avoid filter nn, the second avoids tiny useless expansions
                    # each new neuron adds in_features weights + 1 bias, so we check the real cost before expanding
                    new_neurons = z_score * layer.exp_rate
                    cost = new_neurons * (layer.in_features + 1)
                    if self.model_size() + cost > self.max_params:
                        raise YeastRuntimeError(f"The model reached the max size, or this expansion would break it!")
                    layer.expand(expansion=z_score, weight_template=w_template, bias_template=b_template)
                    self._sync_params(weight)
                    self._sync_params(bias)
        self.optimizer.step()

    def _sync_params(self, parameter: XParameter) -> None:
        """
        This function synchronize the PyTorch optimizer dict, adding the blank space for new parameters data
        :param parameter: the parameter you want to synchronize with its dicts
        :return: none
        """
        if parameter in self.optimizer.state:
            state = self.optimizer.state[parameter]
            for key, value in state.items():
                # noinspection protected-member
                if isinstance(value, torch.Tensor) and value.shape == parameter._shape_before_expansion:
                    new_data = torch.zeros_like(parameter)  # creating the new value with new size
                    old_data_position = tuple(slice(0, dim) for dim in value.shape)
                    new_data[old_data_position] = value  # use the slice to put the old data in their previous place
                    state[key] = new_data  # and updates the new dict

    def _get_xlayers(self) -> list[XLinear]:
        """
        Gets all the expanding nn from the given modules
        :return: a list of them
        """
        return [m for m in self.model.modules() if isinstance(m, XLinear)]

    def model_size(self) -> int:
        """
        Counts the total parameters currently in the optimizer.
        :return: total number of parameters
        """
        return sum(p.numel() for p in self.model.parameters())
