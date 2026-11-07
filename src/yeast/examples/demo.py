"""
demo.py
Authors: Alessio Parolini
This file provide a run-ready demonstration of the features of the project "PyYeast"
"""

import torch.nn

from torch import nn

from torch import optim

from yeast.layers import XAdapter, XLinear

from yeast.optimizer import XOptimWrapper


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.layer1 = nn.Linear(in_features=1, out_features=5)

        self.sell = XLinear(in_features=5, out_features=7, exp_rate=1)

        self.adapter = XAdapter(in_features=7, out_features=10, exp_rate=2)

        self.layer3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.sell(x)

        x = self.relu(x)

        x = self.adapter(x)

        x = self.relu(x)

        x = self.layer3(x)

        return x

def main():

    print("""
    \n\nWelcome to PyYeast!
    Package created by: Alessio Parolini (https://my-code-portfolio-ten.vercel.app/)
    Root:
    GitHub: https://github.com/alessioparolini2011/PyYeast/
    PyPI: https://pypi.org/project/pyyeast/
    I hope I'll find useful my project! Remember to leave a star on GitHub!\n\n""")

    model = MyModule()  # creating the model

    optimizer = optim.Adam(model.parameters(), lr=0.01)  # creating the optimizer

    x_optimizer = XOptimWrapper(module=model, optimizer=optimizer, max_params=int(1e8)) # creating the SENN-compatible optimizer with a limit of 100000 parameters

    criterion = nn.MSELoss()  # choosing the loss function

    input_tensor = torch.rand(100, 1) # creating input

    output_tensor = torch.sin(input_tensor) # a simple math operation to find the output

    for epoch in range(2000):

        optimizer.zero_grad()

        model_out = model(input_tensor)

        loss = criterion(model_out, output_tensor)

        loss.backward()

        optimizer.step()

        print(loss.item())

        if epoch % 10 == 0:

            x_optimizer.expand(expansion=7)  # expanding the expanding layers

            print("\nExpansion done!\n")

            print(x_optimizer.model_size(human=True))  # showing the current size

if __name__ == "__main__":

    main()