"""
demonstration.py
Authors: Alessio Parolini
This file provides a run-ready demonstration of the features of the project "PyYeast"
"""
import torch
from torch import nn, optim

from yeast.nn import XAdapter, XLinear
from yeast.optimizer import XOptimizer

"""
TRAINING STATS
"""
EPOCHS = 600
MAX_PARAMS = int(5e5) # if you use exponential remember to use int() because Python considers it a float
WAIT_BEFORE_EXPANSION = 25 # how many times skips the expansion after it does one, to give the time to the params to train themselves

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(in_features=1, out_features=5)
        self.sell = XLinear(in_features=5, out_features=7, exp_rate=1)
        self.adapter = XAdapter(in_features=7, out_features=10, exp_rate=2)
        self.l2 = nn.Linear(in_features=10, out_features=1)  # you can use a classic linear after XAdapter

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.sell(x))
        x = self.relu(self.adapter(x))
        x = self.l2(x)
        return x


def main():
    torch.manual_seed(42)  # keeping results reproducible
    print("""
    \n\nWelcome to PyYeast!
    Package created by: Alessio Parolini (https://my-code-portfolio-ten.vercel.app/)
    Root:
    GitHub: https://github.com/alessioparolini2011/PyYeast
    PyPI: https://pypi.org/project/pyyeast/
    I hope you'll find my project useful! Remember to leave a star on GitHub!\n\n""")

    model = MyModule()  # creating the model
    optimizer = XOptimizer(
        algorithm=optim.Adam,
        model=model,
        params=model.parameters(),
        max_params=MAX_PARAMS,
        wait_for_expansion=WAIT_BEFORE_EXPANSION,
        lr=0.01,
    )
    criterion = nn.MSELoss()  # choosing the loss function

    # Creating 300 evenly spaced points between -3 and 3 (shape:)
    input_tensor = torch.linspace(-3.0, 3.0, 300).unsqueeze(1)
    output_tensor = torch.sin(3 * input_tensor) + 0.5 * torch.cos(7 * input_tensor)

    print("\nStats before training:\n")
    print(f"l1 (classic): {tuple(model.l1.weight.shape)}")
    print(f"sell (XLinear): {tuple(model.sell.weight.shape)}")
    print(f"adapter (XAdapter): {tuple(model.adapter.weight.shape)}")
    print(f"filter: {tuple(model.adapter.filter.weight.shape)}")
    print(f"l2 (classic): {tuple(model.l2.weight.shape)}")
    start_params = optimizer.model_size()
    print(f"total params: {start_params}")

    print("\nTRAINING!\n")
    initial_loss: float | None = None
    best_loss = float("inf")
    best_epoch = 0
    expansions_count = 0

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        model_out = model(input_tensor)
        loss = criterion(model_out, output_tensor)
        loss_val = loss.item()

        if initial_loss is None:
            initial_loss = loss_val  # saving the very first loss to measure total improvement
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch

        loss.backward()

        total_before = optimizer.model_size()  # save total to detect expansion
        optimizer.xstep()
        total = optimizer.model_size()

        expanded = total != total_before  # check if it grew
        if expanded:
            expansions_count += 1

        if expanded or epoch % 100 == 0:
            total_drop = (1 - loss_val / initial_loss) * 100  # overall reduction since epoch 0
            print(
                f"epoch {epoch:3d} | loss: {loss_val:.5f} | "
                f"total drop: -{total_drop:.2f}% | "
                f"params: {total} | expanded: {expanded}"
            )

    with torch.no_grad():
        model(input_tensor) # a last forward for equalize params
        optimizer.xstep(expand=False)  # only synchronizes params

    final_loss = loss.item()
    final_params = optimizer.model_size()
    overall_reduction = (1 - final_loss / initial_loss) * 100

    print("\nStats after training:\n")
    print(f"sell (XLinear): {tuple(model.sell.weight.shape)}")
    print(f"adapter (XAdapter): {tuple(model.adapter.weight.shape)}")
    print(f"filter: {tuple(model.adapter.filter.weight.shape)}")
    print(f"l2 (classic): {tuple(model.l2.weight.shape)}")
    print(f"\n--- Summary ---")
    print(f"Initial loss:      {initial_loss:.5f}")
    print(f"Final loss:        {final_loss:.5f} (-{overall_reduction:.2f}% overall)")
    print(f"Best loss:         {best_loss:.5f} (at epoch {best_epoch})")
    print(f"Total expansions:  {expansions_count}")
    print(f"Total params:      {start_params} -> {final_params} / {optimizer.max_params} (x{final_params / start_params:.1f} growth)")


if __name__ == "__main__":
    main()
