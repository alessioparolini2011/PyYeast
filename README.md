![PyYeast logo](https://raw.githubusercontent.com/alessioparolini2011/PyYeast/refs/heads/master/assets/yeast_logo.png)

![Python](https://img.shields.io/badge/Python-3.12-FFD43B?style=for-the-badge&logo=python&logoColor=306998)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=red)

## The project

**What's this stuff?**

Good question. `PyYeast` is a Python package built for create and handle *SENN*, aka *Self Expanding Neural Networks*.
A *SENN* can be used to create an **AI model that expand itself during the training**, to improve its capacity. 

I built it because I think SENN are very useful and powerful, so **my goal is to create a package to make easier build and manage them**, also implementing them in already existing NN.

**I hope you'll find my project useful. Remember to leave a star ★!**

## The features 

Here you can find **the resources the package provides**:

from sub-package **`layers`**

+ **XLinear**: a linear layer that can expand his structure by itself ("*X*" stands for "e*X*panding"). It does it in **2 ways**:
    + **out features expanding**: it increases the number of neurons, **improving the layer capacity**
    + **in features expanding**: it automatically detects if the input tensor is change, **adding new weights to process it correctly**

+ **XAdapter**: a XLinear layer, but with a special feature:
    + **the filter layer**: it also has a "hidden" XLinear layer, that takes the output and keep it always identical. Thanks of this, **you can use an XAdapter in a static neural network, avoiding crashes**, because the static after the XAdapter receive always the same sizes tensor.

+ **XOptimWrapper**: a special wrapper for PyTorch optimizers, to "upgrade" them into SENN-compatible ones!
    + given the PyTorch Module and the optimizer associated to it, **XOptimWrapper detects automatically all the expanding layers present within it**.
    + Then, using the given optimizer, every time it expand the layers, it automatically overwrites the dicts of the old parameter with the new ones. **For now, it only overwrites them. In the future, it will be able to keep the old datas, adding only the new weights at the dict**.

## Quickstart

To use this package, you first need **Python 3.12 (or newest)** on your device.
  
Then, run in your terminal:

```bash
pip install pyyeast
```

If you want to see the package in action, run the demonstration file (click [here](https://raw.githubusercontent.com/alessioparolini2011/PyYeast/refs/heads/master/src/yeast/examples/demo.py) to see it) with

```bash
pysenn
```

## Credits

For create this projects, I used the incredible framework [PyTorch](https://pytorch.org/). It gave my the resources to customize my componentes and build from scratch new solutions. 

Thank also, to [Hack Club]([https://hackclub.com/) for give a way to build, meet and play to dev teenagers from all around the world.

## About me

**Hi!** I'm Alessio, a young dev from Verona, Italy 🇮🇹.
CS is my passion, and I'm working to make it my future. 
If you're interested to know me, or you just want to have a talk about projects, development or something else...
See my [GitHub profile](https://github.com/alessioparolini2011) or my [website](https://my-code-portfolio-ten.vercel.app/). Have a good day!