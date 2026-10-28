"""
linear.py
Authors: Alessio Parolini
This file contains the resources to create a XLayer (a self expanding linear layer)
"""

__all__ = ["XLinear"]

import torch

from torch import nn

from torch.nn import functional as f

from pysenn.exceptions import XLayersError, LayersValueError, LayersTypeError


class XLinear(nn.Module):

    """
    This class creates a XLinear, abbreviation of expanding linear (layer).
    This is a special linear layer, because it can expand its structure automatically,
    both for user instructions and for the expansion of previous layers
    """

    def __init__(self, in_features:int, out_features:int | None, exp_rate:int = 1):

        """
        :param in_features: the number of starting inputs (with None it detects it by itself)
        :param out_features: the number of out features (this must be entered!)
        :param exp_rate: the numbers of neurons created for
        """

        super().__init__()

        if in_features is not None and not isinstance(in_features, int):

            raise LayersTypeError("Wrong argument: 'in_features' must be an integer")

        self.in_features = in_features if in_features else 0

        if not out_features:

            raise LayersValueError("Missing argument: 'out_features' cannot be None!")

        if not isinstance(out_features, int):

            raise LayersTypeError("Wrong argument: 'out_features' must be a integer!")

        self.out_features = out_features

        if not isinstance(exp_rate, int):

            raise LayersTypeError("Wrong argument: 'exp_rate' must be an integer!")

        self.weights = nn.Parameter(data=(torch.randn(self.out_features, self.in_features)))

        self._expansion = 0

    def expand(self, expansion: int)-> None:

        """
        Adds the new neurons, making expansion possible.
        :expansion: the expansion is given by the SENN optimizer to choose how much expand the layer
        :return: None
        """

        try:

            self._expansion = expansion # expansion rate calculating (if it's calculated by the SENN optimizer, it considers also layer exp_rate)

            new_neurons = torch.randn(self._expansion, self.in_features, device=self.weights.device, dtype=self.weights.dtype) # new neurons creating

            self.weights = nn.Parameter(data=torch.cat([self.weights.detach(), new_neurons], dim=0)) # add them to the layers weights

            self.out_features += self._expansion # out_features updating

        except ValueError as v:

            raise LayersValueError(f"Value error during a XLayer expansion:\n{v}")

        except TypeError as t:

            raise LayersTypeError(f"Type error during a XLayer expansion:\n{t}")

        except Exception as e:

            raise XLayersError(f"Generic error during a XLayer expansion:\n{e}")

    def forward(self, x):

        """
        Get the input and process it using the layer parameters.
        It auto-adapts itself if the X isn't the same of the precedent epoch (for example: a previous layer expanded itself)
        :param x: the input tensor
        :return: the layer output
        """

        try:

            x_size = x.size()

            out = f.linear(x, self.weights[:(self.out_features - self._expansion)])

            zeros = torch.zeros_like(out[:, :self._expansion])

            self._expansion = 0

            return torch.cat([out, zeros], dim=1)

        except ValueError as v:

            raise LayersValueError(f"Value error during a XLayer forward function:\n{v}")

        except TypeError as t:

            raise LayersTypeError(f"Type error during a XLayer forward function:\n{t}")

        except Exception as e:

            raise XLayersError(f"Generic error during a XLayer forward function:\n{e}")
