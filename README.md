![autoNN logo](./assets/pysenn-logo.png)

![Python](https://img.shields.io/badge/Python-3.12-FFD43B?style=for-the-badge&logo=python&logoColor=306998)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=red)

A Python package, based on PyTorch, to build self-expanding Neural Networks and train them with an optimized process.

I'm building it because **I want to make SENN easier to create and manage**. I'm starting from scratches, for create a really efficient and optimized structure.

## File Structure

```
PySENN/
├── src/
│   └── pysenn/
│       ├── __init__.py
│       ├── __main__.py
│       ├── playground/      # this directory has no __init__.py because hasn't got resources to use, only my practice with PyTorch essentials
│       │   ├── _01_tensors.py
│       │   └── _02_operations.py
│       ├── sell/
│       │   ├── __init__.py
│       │   ├── linear.py       # ExpLayer (SELL)
│       │   ├── filter.py       # FilterLayer
│       │   └── module.py       # ExpModule
│       └── utils/
│           ├── __init__.py
│           └── randomdata.py
├── assets/
├── dist/
├── pyproject.toml
├── uv.lock
└── README.md
```

### ExpLayer (SELL):

**The core layer**  
It is the core of the package: is built starting from a nn.Module (avoiding the nn.Linear limits) to have **a full-customizable structure**.
So I've built a linear layer that can expand his structure, adding new neurons and **improving automatically the neural network power**.
### FilterLayer:

**The bridge**  
A static layer would break if the previous layer grows from 8 to 12 neurons. 
So I've built **FilterLayer**, a layer that takes the variable output of a SELL and **return always the same sizes Tensor**, keeping the flow stable.

### ExpModule:

**The container**   
Groups one ExpLayer + one FilterLayer (putting a ReLU between them) inside a nn.ModuleList. **So you can implement SELL in your nn much easier, because you can use it like a normal layer, it handles everything automatically for you**



## Quickstart

**You need Python 3.12 on your machine**

To install the package, run:

```bash
pip install swell
```

As I showed before, you can use a ExpModule to implement a SELL in your nn:

```python
from swell.sell import ExpModule

module = ExpModule(in_features=10,
                   out_features=22)  # use it like a normal layer, the features inside manage everything for you

module.expand(4)  # add 4 neurons

module(x)  # use it like a normal nn.Module
```

>![WARNING]
>Because of the self-expanding structure, for now **the package doesn't work with optimizers** (optimizers like ADAM have a gradients memory that crashes if you add new). The only way is **using SGD optimizer and recreating it every time you expand the neural network**. See below.

Here's how you should do to use a SGD optimizer:

```python

import torch.optim as optim

from swell.sell import ExpModule

senn = ExpModule(in_features=5, out_features=10)

optimizer = optim.SGD(model.parameters(), lr=0.01)  # model is a fake nn.Module with static and self-expanding layers

...

senn.expand(4)  # expand

optimizer = optim.SGD(model.parameters(), lr=0.01)  # CREATE A NEW OPTIMIZER!

```


**Built with:**
- PyTorch - for all tensor operations, autograd and nn.Module base
- Hack Club - for the Terra program, feedback and the Week 1 ship motivation

Without PyTorch this project would not be possible. Thanks to Hack Club for supporting student-built open source.

## 6. Author

**Alessio Parolini**

Student Developer, Italy

Project: PySENN (formerly autoNN) - Self Expanding Neural Networks

Focus: PyTorch, dynamic architectures, efficient training

GitHub: github.com/alessioparolini2011/ [Alessio Parolini](https://github.com/alessioparolini2011/)

PyPI: pysenn
