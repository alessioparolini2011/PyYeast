"""
xoptim.py
Authors: Alessio Parolini
In this file you can find resources I used to adapt torch.optim.Optimizer to PyYeast componentes
"""

from yeast.exceptions import XOptimizerError, OptimizerValueError, OptimizerTypeError
from yeast.layers import XLinear

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

        self.x_layers_list: list[XLinear] = []  # this list will contain

    def __getattr__(self, item: str) -> Any:
        """
        If the user call .getattr() on the object, it searches also in the original optimizer
        :param item: is the classic argument of getattr
        :return: Any
        """
        return getattr(self.optimizer, item)

    def _get_xlayers(self) -> list:
        """
        Gets all the expanding layers from the given modules
        :return: a list of them
        """
        x_layers: list[XLinear] = []
        for module in self.modules:
            for l in module.parameters():
                if isinstance(l, XLinear):
                    x_layers.append(l)
        return x_layers
