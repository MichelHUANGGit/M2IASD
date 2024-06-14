import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA_Linear(nn.Module):

    def __init__(self, base_layer:nn.Module, r:int):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.base_layer.weight.requires_grad = False
        self.output_dim, self.input_dim = base_layer.weight.shape

        self.B = nn.Linear(in_features=self.input_dim, out_features=r, bias=False)
        self.B.weight = nn.init.zeros_(self.B.weight)
        self.A = nn.Linear(in_features=r, out_features=self.output_dim, bias=False)
        self.A.weight = nn.init.normal_(self.A.weight)

    def forward(self, x:torch.Tensor):
        # print("x", x.shape)
        # print("B", self.B.weight.shape)
        # print("A", self.A.weight.shape)
        y = self.base_layer(x)
        y += self.A(self.B(x))
        return y