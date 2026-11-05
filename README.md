![PyYeast logo](https://raw.githubusercontent.com/alessioparolini2011/PyYeast/refs/heads/master/assets/yeast_logo.png)

![Python](https://img.shields.io/badge/Python-3.12-FFD43B?style=for-the-badge&logo=python&logoColor=306998)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=red)

## The project

**What's this stuff?**

Good question. `PyYeast` is a Python package built for create and handle *SENN*, aka *Self Expanding Neural Networks*.
A *SENN* can be used to create an **AI model that expand itself during the training**, to improve its capacity. 

I built it because I think SENN are very useful and powerful, so **my goal is to create a package to make easier build and manage them**, also implementing them in already existing NN.

All started when I was trying to build my first model using PyTorch, and I noticed that, every time I trained it, the loss would get stuck, at a certain point.
So I thought: <<**There is a way, when the loss freezes, to add a new layer and improve the results?**>> Then I started do some researches on the Internet, discovering **SENN** and understanding better how do they specifically work.

After I learned the basics, I started to build this package, **using low-level interaction with PyTorch components and trying to provide optimized logic and data flow.**

**I hope you'll find my project useful. Remember to leave a star ★!**

## Resources

---

### yeast.layers

#### XLinear

The `XLinear` class provides an easy and quick way to create and use a SELL (Self Expanding Linear Layer)

**Features:**

+ It can expand its structure vertically, increasing **its neurons number**
+ It has got an input detecting function, so **it can expand its number of weights at run-time to adapt for a new input size**, due to a previous layer expansion 
+ Every time it creates new weights (both vertically or inside a single neuron), it initializes them at **0** (or with tiny numbers), avoiding loss spikes in the firsts epochs.

**Initializing**

```python
from yeast.layers import XLinear

sell = XLinear(in_features=10, out_features=22, exp_rate=2)

```

This is how create a XLinear object (it's pretty similar to a classic PyTorch Linear):

+ `in_features: int/None`: the input values in the incoming Tensor (like PyTorch). You can also omit this, the layer will detect it automatically the first epoch

+ `out_features: int`: the number of neurons of the layer, or if you prefer, the number of outputs value in the Tensor the layer calculates (here too, the same as PyTorch).

+ `exp_rate: int`: this value is used in the expansion. When the optimizer gives an expansion rate for the layers, that value is multiplied with this, so every layer can be expanded in a different way

**Usage**

```python
import torch

from yeast.layers import XLinear

sell = XLinear(in_features=10, out_features=22, exp_rate=2)

sell.expand(5)

x = torch.randn(15, 10)  # creating a random input 

output = sell.forward(x)

```

These are the 2 attributes an XLinear instance has:

+ `expand(expansion: int)`: how many neurons add to the layer. This value is multiplied with the `exp_rate` of the layer

+ `forward(x: torch.Tensor)`: works like a PyTorch linear layer forward: get the input tensor and applies the layer's weights and biases to it, returning the output tensor.

#### XAdapter

**Features**

+ `XAdapter` is a double-layer sistem, **consisting of two expanding layer, with A ReLU in the middle**
+ What makes it special is that the second layer, the *Filter Layer*, that **only expand "horizontally", keeping the size of its output tensor always the same**. 
+ In this way, **you can add expanding layers in static neural networks**, because the *Filter Layer* takes care, **all by itself** (you don't need to do anything) to give to the next layer, a static one, **always the same size tensor**

**Initializing**

```python

from yeast.layers import XAdapter

adapter = XAdapter(in_features=15, out_features=20, exp_rate=2)

```

See [XLinear](#xlinear), it's the same

**Usage**

See [XLinear](#xlinear), it's the same

---

### yeast.optimizer

#### XOptimWrapper

**Features**

+ `XOptimWrapper` is a special wrapper for PyTorch optimizers. In fact, **a normal optimizers cannot work with an expanding layer**, because the dicts of the parameters are set when the optimizer itself is created.
+ This class is built to solve this problem: it takes as arguments the model and the related optimizer, **and every time the layers expand, it automatically updates the dicts of the parameters.**
+ It also manages the expansion of all the layers (that it detects by itself when it's created) of the model
>[!WARNING]
> If you're using a `XOptimWrapper` instance, be sure of use only it to expand the layers. Because if you expand a single layer manually, its parameters dicts won't be updated. I'll fix this in the next versions

**Initializing**

```python

from torch import nn, optim

from yeast import optimizer

model = nn.Module()  

optim = optim.Adam(params=model.parameters())

x_optim_wrapper = optimizer.XOptimWrapper(module=model, optimizer=optim)
```

There are two arguments an `XOptimWrapper` needs when it's created:

+ `module: nn.Module`: the model, instance of nn.Module (it requires the entire module because it needs to get the expanding layers from it)
+ `optimizer: optim.Optimizer`: given the optimizer, every time the layers expand themselves, it updates the optimizer dict, **leaving old progress and adding only the blank spaces for new data**. All it's done automatically.

**Usage**

```python

"""
USING THE PRECEDENT SNIPPET RESOURCES
"""

x_optim_wrapper.layers_expand(expansion=3)

```

+ `layers_expansion(expansion: int)`: for every expanding layers in the module passed in the argument, it expands his structure, **like you calling `layer.expand(expansion=4)` for every expanding layer in the module**

>[!WARNING]
> If you use `XOptimWrapper` you've still to do `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()` and all that stuff, **because (for now) the `XOptimWrapper` don't do it by itself**

---

**If you want to some of these resources in action together**, take a look at the demonstration script [here](https://raw.githubusercontent.com/alessioparolini2011/PyYeast/refs/heads/master/src/yeast/examples/demo.py)

## Quickstart

To use this package, you first need **Python 3.12 (or newest)** on your device.
  
Then, run in your terminal:

```bash
pip install pyyeast
```

If you want to see the package in action, run the demonstration file (click [here](https://raw.githubusercontent.com/alessioparolini2011/PyYeast/refs/heads/master/src/yeast/examples/demo.py) to see it) with

```bash
yeast
```

## Credits

For create this projects, I used the incredible framework [PyTorch](https://pytorch.org/). It gave my the resources to customize my componentes and build from scratch new solutions. 

Thank also, to [Hack Club]([https://hackclub.com/) for give a way to build, meet and play to dev teenagers from all around the world.

## About me

**Hi!** I'm Alessio, a young dev from Verona, Italy 🇮🇹.
CS is my passion, and I'm working to make it my future. 
If you're interested to know me, or you just want to have a talk about projects, development or something else...
See my [GitHub profile](https://github.com/alessioparolini2011) or my [website](https://my-code-portfolio-ten.vercel.app/). Have a good day!