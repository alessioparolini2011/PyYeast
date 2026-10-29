![swell logo](./assets/yeast_logo.png)

![Python](https://img.shields.io/badge/Python-3.12-FFD43B?style=for-the-badge&logo=python&logoColor=306998)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=red)
![Numpy](https://img.shields.io/badge/NumPy-2.1-AA0244?style=for-the-badge&logo=numpy&logoColor=white)

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

