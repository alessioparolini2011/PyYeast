"""
filter.py
Authors: Alessio Parolini
This module contains the class for the filter Linear Layer to adapt a SELL to every NN
"""
from .linear import ExpLayer

import torch

from torch.nn import functional as f

from torch import nn

class FilterLayer(ExpLayer):

    def __init__(self, in_features:int, out_features:int):

        super().__init__(in_features=in_features, out_features=out_features)

        self._new_parameters = 0

    def expand(self, expansion:int)-> None:

        """
        Custom version of expand() from ExpLinear
        Adds the new weights in each layer neuron, making expansion possible.
        :param expansion: how many weights you want to add
        :return: None
        """

        self._new_parameters = expansion

        new_weights = torch.randn(self.out_features, expansion)

        self.weights = nn.Parameter(data=torch.cat([self.weights.data, new_weights], dim=1))

    def forward(self, x):

        """
        Custom version of forward() from ExpLinear
        :param x: the input tensor
        :return: the layer output
        """

        self._new_parameters = 0

        return f.linear(x, self.weights)
