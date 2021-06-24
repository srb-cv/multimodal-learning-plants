from torch import Tensor
import torch.nn as nn

from models.fusion import Fusion


class FusionModel(nn.Module):
    def __init__(self,
                 submodels: dict[str, nn.Module],
                 latent_dim: int,
                 out_dim: int):
        super().__init__()
        self.submodels = nn.ModuleDict(submodels)
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.modalities = list(self.submodels.keys())
        self.fusion = Fusion(in_tensors=len(self.modalities),
                             in_features=self.latent_dim,
                             out_features=self.out_dim,
                             bias=True)

    def forward(self, x: dict[str, Tensor]):
        x = self.forward_submodels(x)
        x = self.fusion(x)
        return x

    def forward_submodels(self, x: dict[str, Tensor]):
        return [self.submodels[modality](x[modality]) for modality in self.modalities]

    def modality_scores(self, p):
        return dict(zip(self.modalities, self.fusion.scores(p)))
