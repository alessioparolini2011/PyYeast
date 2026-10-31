"""
wrapper.py
Authors: Alessio Parolini
Contains the resources to make a normal optimizer a SENN-compatible one
"""

from torch.optim import Optimizer

from torch.nn import Module

from yeast.layers import XLinear

from yeast.exceptions import XOptimizerError, OptimizerValueError, OptimizerTypeError

class XOptimWrapper:

    """
    Takes a normal PyTorch optimizer and transform it in a SENN-compatible one,
    that accepts, avoiding errors, new parameters and their state
    """

    def __init__(self, module:Module, optimizer:Optimizer):

        if not isinstance(module, Module):

            raise OptimizerTypeError("Wrong argument: 'module' must be a PyTorch Module or an his sub-class")

        self.module = module

        if not isinstance(optimizer, Optimizer):

            raise OptimizerTypeError("Wrong argument: 'optimizer' must be a PyTorch Optimizer or an his sub-class")

        self.optimizer = optimizer

        self.x_list = []

    def _get_xlayers(self):

        for name, layer in self.module.named_modules():  # get the expanding layers in the module

            if isinstance(layer, XLinear):

                layer_data = {"name" : name,
                              "object" : layer,
                              "exp_rate" : layer.exp_rate}  # in and out features are excluded, due of their elastic structure

                self.x_list.append(layer_data)

    def layers_expand(self, expansion:int):

        if isinstance(expansion, int):

            raise OptimizerTypeError("Wrong argument: 'expansion' must be an integer!")

        if expansion <= 0:

            raise OptimizerValueError("Wrong value: 'expansion' must be higher than 0!")

        try:

            for layer_data in self.x_list:

                layer = layer_data["object"]

                exp_rate = layer_data["exp_rate"]

                layer.expand(expansion=expansion*exp_rate)

        except ValueError as v:

            raise OptimizerValueError(f"Value error during a XLayer expansion by the optimizer:\n{v}")

        except TypeError as t:

            raise OptimizerTypeError(f"Type error during a XLayer expansion by the optimizer:\n{t}")

        except Exception as e:

            raise XOptimizerError(f"Generic error during a XLayer expansion by the optimizer:\n{e}")




