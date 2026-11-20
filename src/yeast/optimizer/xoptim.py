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
from yeast.layers import XLinear
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
                 model: nn.Module | Iterable[nn.Module],
                 params: Iterable[Tensor],
                 max_params: int,
                 **kwargs: Any, ):
        """
        :param algorithm: the optimizer algorithm you choose for the optimizer. Use the same name you find on PyTorch (Adam, SGD...)
        :param model: the PyTorch nn.Module
        :param params: the argument params is the same you'd pass to a classic optimizer
        :param max_params: max total parameters allowed, same limit as XOptimWrapper
        :param kwargs: the others arguments for the optimizer you choose
        """
        if not issubclass(algorithm, Optimizer):
            raise TypeError("Wrong argument: 'algorithm' must be a subclass of PyTorch 'Optimizer'!")
        self.optimizer = algorithm(params=params, **kwargs)  # creates the classic optimizer
        self.max_params = max_params
        if isinstance(model, nn.Module):
            self.modules = [model]
        elif isinstance(model, Iterable):
            self.modules = list(model)
            if not all(isinstance(m, nn.Module) for m in self.modules):
                raise TypeError("Wrong argument: 'model' iterable must contain only PyTorch 'nn.Modules' instance")
        else:
            raise TypeError("Wrong argument: 'model' must be a PyTorch 'nn.Module' instance or an iterable of them")
        self.xlayers_list: list[XLinear] = self._get_xlayers()  # this list is for containing the references to the expanding layers

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


    def xstep(self) -> None:
        """
        This function is designed to expand the layers, using optimized algorithms,
        also updating their dicts (like 'state_dict' or 'param_groups') so the PyTorch optimizer doesn't break
        Also calls the classic .step() method from the original optimizer
        :return: None
        """
        for layer in self.xlayers_list:
            bias = layer.bias
            weight = layer.weight  # saves the two old parameters (they're necessary for get the position in optimizer dicts)
            self._sync_params(weight)
            self._sync_params(bias)  # this operation is SUPER IMPORTANT to synchronize the parameters added during the forward (with _input_adapting) due to a previous layer expansion
            gss = _gss(layer=layer)  # get the tensor with the sum of gradients square
            if gss.numel() < 2:  # if there's only a neuron, skips
                continue
            max_idx = torch.argmax(gss).item()  # get the index of the highest sum
            max_gss = torch.max(gss)  # gets the value of the highest gss
            w_template = layer.weight[max_idx]
            b_template = layer.bias[max_idx: max_idx + 1]  # getting the weight and bias (I used slicing to keep 1D) template (the ones with the highest gradients square sum)
            std, mean = torch.std_mean(gss, correction=0)
            z_score = round(((max_gss - mean) / (
                    std + 1e-8)).item())  # calculating the z-score, it knows how much the highest value deviates from the mean. It's rounded, so it can be used for the expansion
            if layer.exp_rate > 0 and z_score >= 2:  # the first condition avoid filter layers, the second avoids tiny useless expansions
                # each new neuron adds in_features weights + 1 bias, so we check the real cost before expanding
                new_neurons = z_score * layer.exp_rate
                cost = new_neurons * (layer.in_features + 1)
                if self._total_params() + cost > self.max_params:
                    raise YeastRuntimeError(f"The model reached the max size, or this expansion would break it!")
                layer.expand(expansion=z_score, w_template=w_template, b_template=b_template)
                self._sync_params(weight)
                self._sync_params(bias)
        self.optimizer.step()

    def _sync_params(self, parameter: nn.Parameter) -> None:
        """
        This function synchronize the PyTorch optimizer dict, adding the blank space for new parameters data
        :param parameter: the parameter you want to synchronize with its dicts
        :return: none
        """
        for g in self.optimizer.param_groups:
            for i, p in enumerate(g["params"]):
                if p is parameter:
                    if p in self.optimizer.state:  # and updates it in the state dict
                        for key, value in self.optimizer.state[p].items():
                            if isinstance(value, Tensor) and value.size() == p.size():  # in this way, it considers only data about the size of the parameter, the ones is interested in expanding
                                new_data = torch.zeros_like(p)  # creating the new value with new size
                                old_data_position = tuple(slice(0, dim) for dim in p.shape)
                                new_data[old_data_position] = value  # use the slice to put the old data in their previous place
                                self.optimizer.state[key] = new_data  # and updates the new dict
                            else:
                                self.optimizer.state[key] = value  # if the value isn't related to the size, is duplicated

    def _get_xlayers(self) -> list[XLinear]:
        """
        Gets all the expanding layers from the given modules
        :return: a list of them
        """
        x_layers: list[XLinear] = []
        for module in self.modules:  # inspects the modules
            for l in module.modules():
                if isinstance(l, XLinear):  # and gets expanding layers
                    x_layers.append(l)
        return x_layers

    def _total_params(self) -> int:
        """
        Counts the total parameters currently in the optimizer.
        :return: total number of parameters
        """
        return sum(p.numel() for g in self.optimizer.param_groups for p in g["params"])
