"""
_02_tensor.py
This file contains exercises I've done for learn how to really use PyTorch Tensors operations
following documentation at: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
"""

import torch

print("Let's do some operations with Tensors!")

# creating 2 random Tensors with integer values from 0 to 9

a_tensor = torch.randint(0, 10, (2,3), dtype=torch.int8)

b_tensor = torch.randint(0, 10, (2,3), dtype=torch.int8)

print(f"""\nThese are two Tensors:\n
A = {a_tensor}\n
B = {b_tensor}\n""")

print(f"""
\nWe can slice them:\n
First row: A= {a_tensor[0]}, B= {b_tensor[0]}\n
First column: A = {a_tensor[:,0]}, B = {b_tensor[:,0]}""")

print(f"""
We can concat them:\n
{torch.cat([a_tensor, b_tensor], dim=1)}""")

matmul_result = a_tensor @ b_tensor.T # T stands for "transpose" because number of columns in the first must be the same of rows of the second (m, p) @ (p, n) = (m, n)

mul_result = a_tensor * b_tensor  # cause they must have the same shape

print(f"""
There are 2 types of multiplication:\n
@ and *.
The first one does a real matrix multiplication (I transposed the second Tensor):\n
{matmul_result}\n
And the second a element-by-element multiplication:\n
{mul_result}""")

print(f"""
You can also do an simple operations and store the result directly in the operand.\n
Like add the same value to all numbers:\n
{a_tensor.add_(5)}""")

