"""
linear.py
Authors: Alessio Parolini
This file contains the resources to create a XLayer (a self expanding linear layer)
"""

__all__ = ["XLinear"]

import torch

from torch import nn

from torch.nn import functional as f

from yeast.exceptions import XLayersError, LayersValueError, LayersTypeError


class XLinear(nn.Module):

    """
    This class creates a XLinear, abbreviation of expanding linear (layer).
    This is a special linear layer, because it can expand its structure automatically,
    both for user instructions and for the expansion of previous layers
    """

    def __init__(self, out_features:int, in_features:int | None = None,  exp_rate:int = 1):

        """
        :param in_features: the number of starting inputs (with None it detects it by itself)
        :param out_features: the number of out features (this must be entered!)
        :param exp_rate: how much is the extension increased (for the optimizer)
        """

        super().__init__()

        if in_features is not None and not isinstance(in_features, int):

            raise LayersTypeError("Wrong argument: 'in_features' must be an integer")

        self.in_features = in_features if in_features is not None else 0

        if out_features is None:

            raise LayersValueError("Missing argument: 'out_features' cannot be None!")

        if not isinstance(out_features, int):

            raise LayersTypeError("Wrong argument: 'out_features' must be a integer!")

        self.out_features = out_features

        if not isinstance(exp_rate, int):

            raise LayersTypeError("Wrong argument: 'exp_rate' must be an integer!")

        if exp_rate <= 0:

            raise LayersValueError("Wrong value: 'exp_rate' must be higher than zero!")

        self.exp_rate = exp_rate

        self.weights = nn.Parameter(data=(torch.randn(self.out_features, self.in_features)))

        self._expansion = 0

        self._old_parameter_id = None  # to save the deleted parameter and get it (for the optimizer)

    def expand(self, expansion: int)-> None:

        """
        Adds the new neurons, making expansion possible.
        :expansion: the expansion is given by the SENN optimizer to choose how much expand the layer
        :return: None
        """

        try:

            self._expansion = expansion * self.exp_rate # calculate how much expand

            new_neurons = torch.zeros(self._expansion, self.in_features, device=self.weights.device, dtype=self.weights.dtype) # new neurons creating

            self.weights = nn.Parameter(data=torch.cat([self.weights, new_neurons], dim=0)) # add them to the layers weights

            self.out_features = self.weights.size(0) # out_features updating

        except ValueError as v:

            raise LayersValueError(f"Value error during a XLayer expansion:\n{v}")

        except TypeError as t:

            raise LayersTypeError(f"Type error during a XLayer expansion:\n{t}")

        except Exception as e:

            raise XLayersError(f"Generic error during a XLayer expansion:\n{e}")

    def _input_adapting(self, x)-> None:

        """
        Detect if the input tensor is different (because of a previous layer expansion) and adapt the structure
        to avoid crashes.
        :param x: the input tensor
        :return: None
        """

        x_size = x.size(1) # get the number of inputs in the input tensor

        extra_size = x_size - self.in_features # check if are more than the previous epoch

        if extra_size == 0: # if it's not return

            return

        new_param = torch.zeros(self.weights.size(0), extra_size, device=self.weights.device, dtype=self.weights.dtype)

        self.weights = nn.Parameter(torch.cat([self.weights, new_param], dim=1)) # add new weights

        self.in_features = self.weights.size(1) # update the in_features value

    def forward(self, x):

        """
        Get the input and process it using the layer parameters.
        It auto-adapts itself if the X isn't the same of the precedent epoch (for example: a previous layer expanded itself)
        :param x: the input tensor
        :return: the layer output
        """

        try:

            self._input_adapting(x)

            x = f.linear(x, self.weights)

            self._expansion = 0

            return x

        except ValueError as v:

            raise LayersValueError(f"Value error during a XLayer forward function:\n{v}")

        except TypeError as t:

            raise LayersTypeError(f"Type error during a XLayer forward function:\n{t}")

        except Exception as e:

            raise XLayersError(f"Generic error during a XLayer forward function:\n{e}")
