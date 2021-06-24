from typing import Union, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor


class Fusion(nn.Module):
    def __init__(self,
                 in_tensors: int,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super().__init__()
        self.in_tensors = in_tensors
        self.in_features = in_features
        self.out_features = out_features
        self.batch_norm = nn.BatchNorm1d(num_features=in_features * in_tensors, affine=False)
        self.fusion = nn.Linear(in_features=in_features * in_tensors, out_features=out_features, bias=bias)

    @property
    def bias(self):
        return self.fusion.bias

    @property
    def weight(self):
        return self.per_out_weight.transpose(0, 1)  # shape: (in_tensors, out_features, in_features)

    @property
    def per_out_weight(self):
        return self.fusion.weight.reshape(self.out_features, self.in_tensors,  self.in_features)

    def forward(self, tensors: Union[Tuple[Tensor, ...], List[Tensor]]):
        tensors = torch.cat(tensors, dim=1)
        tensors = self.batch_norm(tensors)
        return self.fusion(tensors)

    def regularizer(self, p=1):
        q = 2*p/(p+1)
        return torch.sum(self.weight_norms() ** q) ** (2 / q)

    def scores(self, p=1):
        norms = self.weight_norms()
        a = norms ** (2/(p+1))
        b = torch.sum(norms ** (2*p/(p+1))) ** (1/p)
        return a/b

    def per_out_scores(self):
        out_weight_norms = self.per_out_weight_norms()
        return out_weight_norms / torch.sum(out_weight_norms, dim=1, keepdim=True)

    def weight_norms(self):
        return torch.norm(self.weight, dim=[1, 2])  # shape: (in_tensors)

    def per_out_weight_norms(self):
        return torch.norm(self.per_out_weight, dim=2)  # shape: (out_features, in_tensors)
