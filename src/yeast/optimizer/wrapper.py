"""
wrapper.py
Authors: Alessio Parolini
Contains the resources to make a normal optimizer a SENN-compatible one
"""

__all__ = ["XOptimWrapper"]

from typing import TypedDict

from torch.optim import Optimizer

from torch import nn

from yeast.layers import XLinear, XAdapter

from yeast.exceptions import XOptimizerError, OptimizerValueError, OptimizerTypeError

class LayersDict(TypedDict):

    """
    This special dictionary saves all the expanding layers and their parameters
    """

    layer: XLinear  # the object itself

    bias: nn.Parameter | None

    weight: nn.Parameter | None  # these two keys will save the bias and the weight of the layer, for use them after the expansion

class XOptimWrapper:

    """
    Takes a normal PyTorch optimizer and transform it in a SENN-compatible one,
    that accepts, avoiding errors, new parameters and their state
    """

    def __init__(self, module:nn.Module, optimizer:Optimizer):

        if not isinstance(module, nn.Module):

            raise OptimizerTypeError("Wrong argument: 'module' must be a PyTorch Module or an his sub-class")

        self.module: nn.Module = module

        if not isinstance(optimizer, Optimizer):

            raise OptimizerTypeError("Wrong argument: 'optimizer' must be a PyTorch Optimizer or an his sub-class")

        self.optimizer: Optimizer = optimizer

        self.xl_list: list[LayersDict] = self._get_xlayers()

    def _get_xlayers(self)-> list:

        xl_list = []

        for layer in self.module.modules():  # get the expanding layers in the module

            if isinstance(layer, XLinear):

                layer_dict: LayersDict = {
                    "layer" : layer,
                    "bias" : None,  # empty for now
                    "weight" : None # the same (this keeps the memory free for a little while longer)
                }

                xl_list.append(layer_dict)

        return xl_list

    def layers_expand(self, expansion:int):

        if not isinstance(expansion, int):

            raise OptimizerTypeError("Wrong argument: 'expansion' must be an integer!")

        if expansion <= 0:

            raise OptimizerValueError("Wrong value: 'expansion' must be higher than 0!")

        try:

            for layer_dict in self.xl_list:

                layer = layer_dict["layer"]

                layer_dict["bias"] = layer.bias

                layer_dict["weight"] = layer.weight

                layer.expand(expansion=expansion*layer.exp_rate)

            self._update_dict()

        except ValueError as v:

            raise OptimizerValueError(f"Value error during a XLayer expansion by the optimizer:\n{v}")

        except TypeError as t:

            raise OptimizerTypeError(f"Type error during a XLayer expansion by the optimizer:\n{t}")

        except Exception as e:

            raise XOptimizerError(f"Generic error during a XLayer expansion by the optimizer:\n{e}")


    def _update_dict(self):

        """
        Overwrites the old dict of a parameter (both in 'state' and 'param_group').
        For now, only overwrite (losing progress). In the future, it will keep the old datas, adding only the new ones
        :return: None
        """

        for layer_dict in self.xl_list:

            old_bias = layer_dict["bias"]

            old_weight = layer_dict["weight"]

            for group in self.optimizer.param_groups:

                for i, p in enumerate(group["params"]):

                    if p is old_bias or p is old_weight:

                        group["params"][i] = layer_dict["layer"].bias if p is old_bias else layer_dict["layer"].weight  # replaces the old parameter

                        self.optimizer.state.pop(p,None)  # deletes the old parameter state dict

                        self.optimizer.state[group["params"][i]] = {}  # overwrite the parameter state dict - the lazy initialization will set it automatically
