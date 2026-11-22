"""
adapter.py
Authors: Alessio Parolini
Contains the class that groups ExpLinear and FilterLayer for an easier implementation
"""

from torch.nn import functional as f

from yeast.nn.xlinear import XLinear


class XAdapter(XLinear):
    """
    This class is designed to create XLinear nn that can be inserted in a classic NN without input sizes errors.
    For ensure this, the XAdapter layer has a second "filter" XLinear layer that keep the output identical during all the training
    """

    def __init__(self, in_features: int, out_features: int, exp_rate: int = 1):
        """
        The passed arguments are the same for XLinear
        """
        super().__init__(in_features=in_features, out_features=out_features, exp_rate=exp_rate)
        self.filter = XLinear(in_features=self.out_features, out_features=self.out_features)
        self.filter.exp_rate = 0  # "exp_rate" is set to zero, so if the XOptimWrapper expand all the XLinear instance, the filter always keep his structure the same. I've done it after the creation to avoid expections

    def forward(self, x):
        """
        It's the same of XLinear, only adds the ReLU in the middle and the filter layer
        """
        x = f.relu(super().forward(x))
        x = self.filter(x)
        return x
