"""
wrapper.py
Authors: Alessio Parolini
Contains the resources to make a normal optimizer a SENN-compatible one
"""

from typing import TypedDict

import torch

from torch.optim import Optimizer

from torch import nn

from yeast.layers import XLinear

from yeast.exceptions import XOptimizerError, OptimizerValueError, OptimizerTypeError

class LayersDict(TypedDict):

    """
    This special dictionary saves all the expanding layers and their parameters
    """

    layer: XLinear

    bias_dict: dict | None

    weight_dict: dict | None

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

        self.xl_list = self._get_xlayers()

    def _get_xlayers(self)-> list:

        xl_list = []

        for layer in self.module.modules():  # get the expanding layers in the module

            if isinstance(layer, XLinear):

                layer_data: LayersDict = {
                    "layer" : layer,
                    "bias_dict" : None,  # empty for now
                    "weight_dict" : None # the same
                }

                xl_list.append(layer_data)

        return xl_list

    def layers_expand(self, expansion:int):

        if isinstance(expansion, int):

            raise OptimizerTypeError("Wrong argument: 'expansion' must be an integer!")

        if expansion <= 0:

            raise OptimizerValueError("Wrong value: 'expansion' must be higher than 0!")

        try:

            for layer_dict in self.xl_list:

                layer = layer_dict["layer"]

                layer_dict["bias_dict"] = self.optimizer.state.pop(layer.bias, {})

                layer_dict["weight_dict"] = self.optimizer.state.pop(layer.weight, {})

                layer.expand(expansion=expansion*layer.exp_rate)

        except ValueError as v:

            raise OptimizerValueError(f"Value error during a XLayer expansion by the optimizer:\n{v}")

        except TypeError as t:

            raise OptimizerTypeError(f"Type error during a XLayer expansion by the optimizer:\n{t}")

        except Exception as e:

            raise XOptimizerError(f"Generic error during a XLayer expansion by the optimizer:\n{e}")


    def _update_dict(self, new_parameter:nn.Parameter, old_parameter:nn.Parameter):

        pass










