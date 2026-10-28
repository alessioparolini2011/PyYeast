"""
module.py
Authors: Alessio Parolini
Contains the class that groups ExpLinear and FilterLayer for an easier implementation
"""

from torch import nn

from torch.nn import functional as f

from .linear import ExpLayer

from .filter import FilterLayer

class ExpModule(nn.Module):

    def __init__(self, in_features: int, out_features:int):

        super().__init__()

        self.sell = ExpLayer(in_features, in_features)

        self.filter_layer = FilterLayer(in_features, out_features)

        self.structure = nn.ModuleList([self.sell, self.filter_layer])

    def expand(self, expansion: int)-> None:

        """
        Expand the whole structure
        :param expansion: a parameter to specify expansion. If None uses the ExpLinear default
        :return: None
        """

        expansion = expansion if expansion else None

        for module in self.structure:

            module.expand(expansion=expansion)

    def forward(self, x):

        """
        Process the x in the SELL structure (+ ReLU)
        :param x:
        :return:
        """

        for module in self.structure:

            x = module(x)

            x = f.relu(x)

        return x
