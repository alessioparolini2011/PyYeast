<p align="center">
  <img src="https://raw.githubusercontent.com/alessioparolini2011/PyYeast/refs/heads/master/assets/yeast_logo.png" width="400" alt="PyYeast logo">
  <br>
  <img src="https://img.shields.io/badge/Python-3.12-FFD43B?style=for-the-badge&logo=python&logoColor=306998" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hack%20Club-Terra-191970?style=for-the-badge&logo=hackclub&logoColor=white" alt="Hack Club">
</p>

**LATEST RELEASE →
[See on PyPI](https://pypi.org/project/pyyeast/) ←** 

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

### yeast.nn

#### XLinear

The `XLinear` class provides an easy and quick way to create and use a SELL (Self Expanding Linear Layer)

**Features:**

+ It can expand its structure vertically, increasing **its neurons number**
+ It has got an input detecting function, so **it can expand its number of weights at run-time to adapt for a new input size**, due to a previous layer expansion
+ Every time it creates new weights (both vertically or inside a single neuron), it initializes them with **tiny random numbers**, avoiding loss spikes in the firsts epochs.
+ With `exp_rate`, **every layer can grow in a different way**: a layer with a higher `exp_rate` grows faster than the others, so you can control which part of the model expands more
+ It can also expand starting from a template: **you can pass a weight and a bias to copy**, and the new neurons will start as noisy copies of them, instead of starting from scratch
+ It's built on `XParameter`, so **it grows without changing its identity**, and the optimizer keeps tracking the same object

**Initializing**

```python
from yeast.nn import XLinear

sell = XLinear(out_features=22, in_features=10, exp_rate=2)

```

This is how create a XLinear object (it's pretty similar to a classic PyTorch Linear):

+ `out_features: int`: the number of neurons of the layer, or if you prefer, the number of outputs value in the Tensor the layer calculates (here too, the same as PyTorch).

+ `in_features: int`: the input values in the incoming Tensor (like PyTorch). It is required, it is not inferred.

+ `exp_rate: int`: this value is used in the expansion. When the optimizer gives an expansion rate for the layers, that value is multiplied with this, so every layer can be expanded in a different way

**Usage**

```python
import torch

from yeast.nn import XLinear

sell = XLinear(out_features=22, in_features=10, exp_rate=2)

sell.expand(5, weight_template=None, bias_template=None)

x = torch.randn(15, 10)  # creating a random input 

output = sell.forward(x)

```

These are the 2 attributes an XLinear instance has:

+ `expand(expansion: int, weight_template: torch.Tensor | None, bias_template: torch.Tensor | None)`: how many neurons add to the layer. This value is multiplied with the `exp_rate` of the layer. If you pass the templates, the new neurons start as noisy copies of them, if you pass `None`, they start from tiny random numbers.

+ `forward(x: torch.Tensor)`: works like a PyTorch linear layer forward: get the input tensor and applies the layer's weights and biases to it, returning the output tensor. If a previous layer expanded, it adapts its weights by itself before calculating.

#### XAdapter

**Features**

+ `XAdapter` is a double-layer sistem, **consisting of two expanding layer, with A ReLU in the middle**

+ What makes it special is that the second layer, the *Filter Layer*, that **only expand "horizontally", keeping the size of its output tensor always the same**.

+ In this way, **you can add expanding layers in static neural networks**, because the *Filter Layer* takes care, **all by itself** (you don't need to do anything) to give to the next layer, a static one, **always the same size tensor**

+ **No manual reshaping:** you can put a classic `nn.Linear` after an `XAdapter` and it will never break, even if the adapter grows inside

+ Its *Filter Layer* has `exp_rate = 0`, so **when you expand all the layers of the model, the filter never grows vertically**, it only adapts to the new input size

**Initializing**

```python

from yeast.nn import XAdapter

adapter = XAdapter(in_features=15, out_features=20, exp_rate=2)

```

See [XLinear](#xlinear), it's the same

**Usage**

See [XLinear](#xlinear), it's the same

#### XParameter

The `XParameter` class provides a special version of PyTorch `nn.Parameter` that can grow at run-time

**Features:**

+ It's a subclass of `nn.Parameter`, so **it works exactly like a normal PyTorch parameter**
+ It can add new params at run-time, **without changing its identity**. In this way, the optimizer keeps tracking the same object, even after an expansion
+ It updates its `.data` in place, so **you don't need to recreate the parameter or the optimizer** every time a layer expands
+ If it already has a gradient, **it pads it with zeros** for the new params, so the old gradient is not lost and the new neurons start clean
+ It only stores and grows the data: **splitting and scaling of new parameters is handled by the layer**, not here

**Initializing**

```python
import torch

from yeast.nn import XParameter

param = XParameter(data=torch.randn(10, 5))

```

This is how create a XParameter object (it's pretty similar to a classic PyTorch Parameter):

+ `data: torch.Tensor | None`: the starting values of the parameter (like PyTorch).

+ `requires_grad: bool`: if the parameter needs a gradient or not (like PyTorch).

**Usage**

```python
import torch

from yeast.nn import XParameter

param = XParameter(data= torch.randn(10, 5))

new_params = torch.randn(3, 5) * 1e-2

param.add_params(new_params, dim=0)

```

This is the attribute an XParameter instance has:

+ `add_params(parameters: torch.Tensor, dim: int = 0)`: adds the new params to the parameter, on the dimension you choose. It checks that the sizes are compatible, it saves the old shape, and it updates `.data` in place.

#### XOptimizer

**Features**

+ `XOptimizer` is a special version of PyTorch optimizers. In fact, **a normal optimizer cannot work with an expanding layer**, because the dicts of the parameters are set when the optimizer itself is created.

+ This class is built to solve this problem: you don't create the optimizer outside, and then you use a wrapper, **instead you create the PyTorch optimizer and the SENN-compatible version with the same object.**

+ It also manages the expansion of all the layers (that it detects by itself when it's created) of the model, **using the GSS of every neuron**. Every time it does a step, it checks if a layer needs to expand, and if yes, it expands it and **automatically updates the dicts of the parameters.**

+ **No reset, no lost progress:** when it expands a layer, it keeps the momentum of the old neurons exactly as it was, and the new neurons start with zero momentum. Nothing is restarted from scratch

+ **It skips what must not grow:** it skips the *Filter Layers* of the `XAdapter`, because they have `exp_rate = 0`

+ **It only expands when it really needs it:** it calculates the z-score of the GSS for every layer, and it expands only if the best neuron is really far from the mean. In this way, **it avoids tiny useless expansions**

+ **It never breaks the limit:** before expanding, it calculates the real cost of the new neurons, and if the model would go over `max_params`, it stops instead of saturating the memory

+ **It works like a classic optimizer:** it has the same `param_groups`, `state`, `step()`, `zero_grad()`, `state_dict()` and `load_state_dict()` of a PyTorch optimizer, so you can use it in the same way

>[!WARNING]
> If you're using a `XOptimizer` instance, be sure to use only it to expand the layers and to do the step. Don't call `optimizer.step()` by yourself, because `xstep()` already does it. Because if you expand a single layer manually, its parameters dicts won't be updated.

**Initializing**

```python

from torch import nn, optim

from yeast.nn import XLinear
from yeast import optimizer

model = nn.Sequential(
    XLinear(out_features=16, in_features=10),
    nn.ReLU(),
    XLinear(out_features=1, in_features=16)
)

x_optimizer = optimizer.XOptimizer(algorithm=optim.Adam, model=model, params=model.parameters(), max_params=1000000, wait_for_expansion=5, lr=0.01)

```

There are four arguments an `XOptimizer` needs when it's created, plus the ones of the classic optimizer:

+ `algorithm: type[Optimizer]`: the optimizer algorithm you choose for the optimizer. Use the same name you find on PyTorch (`Adam`, `SGD`...). It must be a subclass of PyTorch `Optimizer`.

+ `model: nn.Module`: the model, instance of nn.Module (it requires the entire module because it needs to get the expanding layers from it)

+ `params: Iterable[Tensor]`: the argument `params` is the same you'd pass to a classic optimizer

+ `max_params: int`: this value set a limit to the model size, so the expansion can't cause a memory saturation. It must be higher than the actual size of the model.

+ `wait_for_expansion: int`: how many times it skips the expansion after it has done one. In this way, the model has time to use the new neurons before growing again.

+ `**kwargs: Any`: the others arguments for the optimizer you choose, like `lr`, `weight_decay`... They are passed directly to the classic optimizer.

**Usage**

```python

"""
USING THE PRECEDENT SNIPPET RESOURCES
"""

# in your training loop, use xstep() instead of step()
x_optimizer.xstep()
x_optimizer.zero_grad()

print(x_optimizer.model_size()) # OUTPUT EXAMPLE: 257

```

+ `xstep(expand=True)`: does the classic `step()` of the optimizer, **and** it checks if a layer needs to expand. If yes, it expands it starting from the neuron with the highest GSS, as a noisy copy of it, and it updates all the optimizer dicts by itself. Argument `expand` is `True` by default. If it's `False`, it only synchronizes the params dicts and does `step()`

+ `step(closure=None)`: works like a classic PyTorch optimizer `step()`. You don't need it if you use `xstep()`, it's there only to keep the compatibility with PyTorch.

+ `zero_grad(set_to_none=True)`: works like a classic PyTorch optimizer `zero_grad()`, it clears the gradients.

+ `model_size()`: return the number of parameters (understood as the individual weights and biases) of the model, as a pure number (*257*, *123*...)

+ `state_dict() / load_state_dict(state_dict)`: work like the classic PyTorch ones, they save and load the state of the optimizer.

>[!WARNING]
> If you use `XOptimizer` you've still to do `loss.backward()` and `zero_grad()` and all that stuff, **because `xstep()` does the `step()` and the expansion, but it doesn't do the backward by itself**

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