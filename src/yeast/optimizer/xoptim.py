"""
xoptim.py
Authors: Alessio Parolini
In this file you can find resources I used to adapt torch.optim.Optimizer to PyYeast componentes
"""
import torch

from yeast.exceptions import OptimizerTypeError
from yeast.layers import XLinear
from ._optim_utils import _gss

from typing import Iterable, Any

from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module


class XOptimizer:
    """
    This class is an improved version of the previous adapter of the PyTorch optimizers, XOptimWrapper.
    The different is that now you don't create the optimizer outside, and then you use a wrapper,
    instead you create the PyTorch optimizer and the SENN-compatible version with the same object
    """

    def __init__(self, algorithm: type[Optimizer],
                 model: Module | Iterable[Module],
                 params: Iterable[Tensor],
                 **kwargs: Any, ):
        """
        :param algorithm: the optimizer algorithm you choose for the optimizer. Use the same name you find on PyTorch (Adam, SGD...)
        :param model: the PyTorch nn.Module
        :param params: the argument params is the same you'd pass to a classic optimizer
        :param kwargs: the others arguments for the optimizer you choose
        """
        if not issubclass(algorithm, Optimizer):
            raise OptimizerTypeError("Wrong argument: 'algorithm' must be a subclass of PyTorch 'Optimizer'!")
        self.optimizer = algorithm(params=params, **kwargs)  # creates the classic optimizer
        if isinstance(model, Module):
            self.modules = [model]
        elif isinstance(model, Iterable):
            self.modules = list(model)
            if not all(isinstance(m, Module) for m in self.modules):
                raise OptimizerTypeError(
                    "Wrong argument: 'model' iterable must contain only PyTorch 'Modules' instance")
        else:
            raise OptimizerTypeError(
                "Wrong argument: 'model' must be a PyTorch 'Module' instance or an iterable of them")
        self.xlayers_list: list[XLinear] = self._get_xlayers()  # this list is for containing the references to the expanding layers
        self._register_hooks()


    def __getattr__(self, item: str) -> Any:
        """
        If the user call .getattr() on the object, it searches also in the original optimizer
        :param item: is the classic argument of getattr
        :return: Any
        """
        return getattr(self.optimizer, item)

    def xstep(self) -> None:
        """
        This function is designed to expand the layers, using optimized algorithms,
        also updating their dicts (like 'state_dict' or 'param_groups') so the PyTorch optimizer doesn't break
        Also calls the classic .step() method from the original optimizer
        :return: None
        """
        for layer in self.xlayers_list:
            old_bias = layer.bias
            old_weight = layer.weight  # saves the two old parameters (they're necessary for get the position in optimizer dicts)
            gss = _gss(layer=layer)  # get the list of the sum of gradients square
            if gss.numel() < 2: # if there's only a neuron, skips
                continue
            max_idx = torch.argmax(gss).item()  # get the index of the highest sum
            w_template = layer.weight[max_idx]
            b_template = layer.bias[max_idx : max_idx + 1]  # getting the weight and bias (I used slicing to keep 1D) template (the ones with the highest gradients square sum)
            std, mean = torch.std_mean(gss, correction=0)
            max_gss = torch.max(gss)
            z_score = round(((max_gss - mean ) / (std + 1e-8)).item()) # calculating the z-score, it knows how much the highest value deviates from the mean. It's rounded, so it can be used for the expansion
            if layer.exp_rate > 0 and z_score >= 2:  # the first condition avoid filter layers, the second avoids tiny useless expansions
                layer.expand(expansion=z_score, w_template=w_template, b_template=b_template)
                self._sync_param(old_weight, layer.weight)
                self._sync_param(old_bias, layer.bias)

        self.optimizer.step()

    def _get_xlayers(self) -> list[XLinear]:
        """
        Gets all the expanding layers from the given modules
        :return: a list of them
        """
        x_layers: list[XLinear] = []
        for module in self.modules:
            for l in module.modules():
                if isinstance(l, XLinear):
                    x_layers.append(l)
        return x_layers

    def _register_hooks(self) -> None:
        for layer in self.xlayers_list:
            def pre_hook(mod: XLinear, inputs: tuple[Tensor, ...]) -> None:
                if not inputs:
                    return
                x = inputs[0]
                if x.size(1) > mod.in_features:
                    old_weight = mod.weight
                    mod._input_adapting(x)
                    if old_weight is not mod.weight:
                        self._sync_param(old_weight, mod.weight)

            layer.register_forward_pre_hook(pre_hook)

    def _sync_param(self, old_p: Tensor, new_p: Tensor) -> None:
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if p is old_p:
                    group["params"][i] = new_p
                    if old_p in self.optimizer.state:
                        old_state = self.optimizer.state[old_p]
                        new_state = {}
                        for key, value in old_state.items():
                            if isinstance(value, torch.Tensor) and old_p.shape == value.shape:
                                new_dims = torch.zeros(new_p.shape, device=new_p.device, dtype=new_p.dtype)
                                data = tuple(slice(0, dim) for dim in old_p.shape)
                                new_dims[data] = value
                                new_state[key] = new_dims
                            else:
                                new_state[key] = value
                        self.optimizer.state[new_p] = new_state
                        del self.optimizer.state[old_p]
                    return

