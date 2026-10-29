"""
adapter.py
Authors: Alessio Parolini
Contains the class that groups ExpLinear and FilterLayer for an easier implementation
"""

import torch

from torch import nn

from torch.nn import functional as f

from .linear import XLinear

from yeast.exceptions import XLayersError, LayersValueError, LayersTypeError

class XAdapter(XLinear):

    """
    This class is designed to create XLinear layers that can be inserted in a classic NN without input sizes errors.
    For ensure this, the XAdapter layer has a second "filter" XLinear layer that keep the output identical during all the training
    """

    def __init__(self, in_features: int | None, out_features:int, exp_rate:int = 1):

        """
        The passed arguments are the same for XLinear
        """

        super().__init__(in_features=in_features, out_features=out_features, exp_rate=exp_rate)

        self.weights = nn.Parameter(data=torch.randn(self.out_features, self.in_features))

        self.filter = XLinear(out_features=self.out_features) #auto-detect the input the first time


    def forward(self, x):

        """
        It's the same of XLinear, only adds the ReLU in the middle and the filter layer
        """

        try:

             x = f.linear(x, self.weights)

             x = f.relu(x)

             x = self.filter.forward(x)

             return x

        except ValueError as v:

            raise LayersValueError(f"Value error during a XLayer forward function:\n{v}")

        except TypeError as t:

            raise LayersTypeError(f"Type error during a XLayer forward function:\n{t}")

        except Exception as e:

            raise XLayersError(f"Generic error during a XLayer forward function:\n{e}")
