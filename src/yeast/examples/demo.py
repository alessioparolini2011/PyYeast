"""
demonstration.py
Authors: Alessio Parolini
This file provide a run-ready demonstration of the features of the project "PyYeast"
"""
import torch
from torch import nn
from torch import optim

from yeast.layers import XAdapter, XLinear
from yeast.optimizer import XOptimizer


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(in_features=1, out_features=5)
        self.sell = XLinear(in_features=5, out_features=7, exp_rate=1)
        self.adapter = XAdapter(in_features=7, out_features=10, exp_rate=2)
        self.l2 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.sell(x))
        x = self.relu(self.adapter(x))
        x = self.l2(x)  # you can use a linear after a
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
    optimizer = XOptimizer(algorithm=optim.Adam, model=model, params=model.parameters(), max_params=int(2e7), wait_for_expansion=25, lr=0.01)  # creating the optimizer
    criterion = nn.MSELoss()  # choosing the loss function
    input_tensor = torch.rand(100, 1)  # creating input
    output_tensor = torch.sin(input_tensor)  # a simple math operation to find the output
    print("\nStats before training:\n")
    print(f"l1 (classic): {tuple(model.l1.weight.shape)}")
    print(f"sell (XLinear): {tuple(model.sell.weight.shape)}")
    print(f"adapter (XAdapter): {tuple(model.adapter.weight.shape)}")
    print(f"filter: {tuple(model.adapter.filter.weight.shape)}")
    print(f"l2 (classic): {tuple(model.l2.weight.shape)}")
    total = sum(p.numel() for p in model.parameters())  # counting total params
    print(f"total params: {total}")
    print("\nTRAINING!\n")
    for epoch in range(700):
        optimizer.zero_grad()
        model_out = model(input_tensor)
        loss = criterion(model_out, output_tensor)
        loss.backward()
        total_before = sum(p.numel() for p in model.parameters())  # save total to detect expansion
        optimizer.xstep()
        total = sum(p.numel() for p in model.parameters())
        expanded = total != total_before  # check if it grew
        if expanded or epoch % 200 == 0:
            with torch.no_grad():
                loss_after = criterion(model(input_tensor), output_tensor)  # loss after expansion
            lost = ((loss_after - loss) / loss * 100).item()  # loss lost in percent for this expansion
            print(f"epoch {epoch} - loss: {loss.item():.4f} - loss after: {loss_after.item():.4f} - lost: {lost:.2f}% - total params: {total} - expanded: {expanded}")
        else:
            print(loss.item())
    print("\nStats after training:\n")
    print(f"sell (XLinear): {tuple(model.sell.weight.shape)}")
    print(f"adapter (XAdapter): {tuple(model.adapter.weight.shape)}")
    print(f"filter: {tuple(model.adapter.filter.weight.shape)}")
    print(f"l2 (classic): {tuple(model.l2.weight.shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"total params: {total} / max_params: {optimizer.max_params}")
    total = sum(p.numel() for p in model.parameters())
    print(f"total params after final_forward: {total}")


if __name__ == "__main__":
    main()