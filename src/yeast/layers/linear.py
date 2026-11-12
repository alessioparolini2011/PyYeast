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

    def __init__(self, out_features: int, in_features: int | None = None, exp_rate: int = 1,):

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

        self.weight = nn.Parameter(data=(torch.randn(self.out_features, self.in_features)))

        self.bias = nn.Parameter(data=(torch.randn(self.out_features)))

        self._last_expansion = 0

    def expand(self, expansion: int,
               w_template: torch.Tensor | None = None,
               b_template: torch.Tensor | None = None,) -> None:
        """
        This function increase the number of neurons of the layer, adding new rows of weights and biases in the parameters.
        It can start from templates w and b, or use random numbers
        :param expansion: how much expand the model
        :param w_template: the template for w
        :param b_template: the template for b
        :return: none
        """
        if not isinstance(expansion, int):
            raise LayersTypeError("Wrong argument: 'expansion' must be an integer!")
        if not expansion > 0:
            raise LayersValueError("Wrong argument: 'expansion' must be higher than 0!")
        try:
            self._last_expansion = expansion * self.exp_rate
            if w_template is not None:
                if not isinstance(w_template, torch.Tensor):
                    raise LayersTypeError("Wrong argument: 'w_template' must be a torch.Tensor instance!")
                if w_template.size() != (self.in_features,):
                    raise LayersValueError("Error! The size of 'w_template' is different by the input features of the layer!")
                new_weight = w_template.to(device=self.weight.device,
                                           dtype=self.weight.dtype).unsqueeze(0).repeat(self._last_expansion, 1)  # creating new weights from the template
                new_weight = new_weight + torch.randn_like(new_weight) * 1e-2  # and adding some noise
            else:
                new_weight = torch.rand(self._last_expansion,
                                        self.in_features,
                                        device=self.weight.device,
                                        dtype=self.weight.dtype) * 1e-2
            if b_template is not None:
                if not isinstance(b_template, torch.Tensor):
                    raise LayersTypeError("Wrong argument: 'b_template' must be a torch.Tensor instance!")
                if b_template.size() != (1,):
                    raise LayersValueError("Error! The size of 'b_template' is different by the output features of the layer!")
                new_bias = b_template.to(device=self.bias.device,
                                         dtype=self.bias.dtype).repeat(self._last_expansion)  # creating new biases
                new_bias = new_bias + torch.randn_like(new_bias) * 1e-2  # and adding some noise
            else:
                new_bias = torch.rand(self._last_expansion,
                                      device=self.bias.device,
                                      dtype=self.bias.dtype) * 1e-2
            self.weight = nn.Parameter(data=torch.cat([self.weight, new_weight],
                                                      dim=0))  # adding the new weights
            self.bias = nn.Parameter(data=torch.cat([self.bias, new_bias],
                                                    dim=0))  # and new biases
            self.out_features = self.weight.size(0)  # out features expanding
        except ValueError as v:
            raise LayersValueError(f"Value error during a XLayer expansion:\n{v}")
        except TypeError as t:
            raise LayersTypeError(f"Type error during a XLayer expansion:\n{t}")
        except Exception as e:
            raise XLayersError(f"Generic error during a XLayer expansion:\n{e}")

    def _input_adapting(self, x) -> None:

        """
        Detect if the input tensor is different (because of a previous layer expansion) and adapt the structure
        to avoid crashes.
        :param x: the input tensor
        :return: None
        """

        x_size = x.size(1)  # get the number of inputs in the input tensor

        extra_size = x_size - self.in_features  # check if are more than the previous epoch

        if extra_size == 0:  # if it's not return

            return

        new_param = torch.zeros(self.weight.size(0), extra_size, device=self.weight.device, dtype=self.weight.dtype)

        self.weight = nn.Parameter(torch.cat([self.weight, new_param], dim=1))  # add new weight

        self.in_features = self.weight.size(1)  # update the in_features value

    def forward(self, x):

        """
        Get the input and process it using the layer parameters.
        It auto-adapts itself if the X isn't the same of the precedent epoch (for example: a previous layer expanded itself)
        :param x: the input tensor
        :return: the layer output
        """

        try:

            self._input_adapting(x)

            x = f.linear(x, self.weight, self.bias)

            self._last_expansion = 0

            return x

        except ValueError as v:

            raise LayersValueError(f"Value error during a XLayer forward function:\n{v}")

        except TypeError as t:

            raise LayersTypeError(f"Type error during a XLayer forward function:\n{t}")

        except Exception as e:

            raise XLayersError(f"Generic error during a XLayer forward function:\n{e}")
