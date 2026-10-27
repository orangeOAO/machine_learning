import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, in_dim= 1, out_dim= 1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.lin(x)
model = Model(1, 1)
print(model)