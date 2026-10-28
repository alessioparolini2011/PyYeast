"""
randomdata.py
Contains the function to generate random datas for test trainings
"""

import torch

__all__ = ["random_xor"]


def random_xor(values: int = 2)-> tuple[torch.Tensor, torch.Tensor]:

    """
    Generates a random tensor and use it to create an XOR pattern.
    Has more data than the used ones for a real training experience.
    :return: the two tensors
    """

    x = torch.randn(1000, 20)

    y = ((x[:, :values] > 0).sum(dim=1) % 2 == 1).float()

    return x, y.unsqueeze(1)