"""
_01_tensor.py
This file contains exercises I've done for learn how to really use PyTorch Tensors
following documentation at: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
"""
import numpy as np
import torch

print("Playground: 01 Tensors and datas\n")

# creating a list and converting it into a tensor

data = [[1,2],[3,4]]

data_tensor = torch.tensor(data)

print(f"\nThis is a tensor: \n{data_tensor}\n")

# creating a numpy array and converting it into a tensor

np_data = np.array([[1,2],[3,4],[5,6]])

np_tensor = torch.from_numpy(np_data)

print(f"\nThis is a Numpy array: \n{np_data}\n and this is a tensor from a that Numpy array: \n{np_tensor}\n")

# creating two tensor with ones and random numbers

ones_tensor = torch.ones_like(data_tensor) # gets the properties from data_tensor

rand_tensor = torch.rand_like(data_tensor, dtype=torch.float) # this tensor has also a specific data type

print(f"\nThis is a ones tensor: \n{ones_tensor}.\nThis is instead a random generated one: \n{rand_tensor}\nThey are both generated from the properties of the first tensor.\n")

# creating tensors, not from an existing one

shape = (3, 2)

new_rand_ten = torch.rand(shape)

new_zeros_ten = torch.zeros(shape)

new_ones_ten = torch.ones(shape)

print(f"\nThis is a random tensor created without starting form an existing one: \n{new_rand_ten}\nThe same, but with zeros: \n{new_zeros_ten}\nand ones: \n{new_ones_ten}\n")

print(f"\nThese are some of the attributes of the first tensor:\nThe shape: {data_tensor.shape}\nThe data type: {data_tensor.dtype}\n And is stored on: {data_tensor.device}\n")
