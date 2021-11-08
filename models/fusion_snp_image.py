from torch import Tensor
import torch.nn as nn

from models.fusion import Fusion
from typing import Dict
import torch


class FusionSNPIMageModel(nn.Module):
    def __init__(self,
                 submodels_image: Dict[str, nn.Module],
                 submodels_snp: Dict[str, nn.Module],
                 latent_dim_image: int,
                 latent_dim_snp: int,
                 out_dim: int):
        super().__init__()
        self.submodel_image = nn.ModuleDict(submodels_image)
        self.submodel_snp = nn.ModuleDict(submodels_snp)
        self.latent_dim_image = latent_dim_image
        self.latent_dim_snp = latent_dim_snp
        self.out_dim = out_dim
        self.modalities_image = list(self.submodel_image.keys())
        self.modalities_snp = list(self.submodel_snp.keys())
        num_features = self._get_num_features()
        self.batch_norm = nn.BatchNorm1d(num_features=num_features,affine=False)
        self.linear = nn.Linear(in_features=num_features,out_features=self.out_dim,bias=True)

    def _get_num_features(self):
        return len(self.submodel_image) * self.latent_dim_image + \
             len(self.submodel_snp) * self.latent_dim_snp


    def forward(self, x: Dict[str, Tensor]):
        x = self.forward_submodels(x)
        x = torch.cat(x, dim=1)
        x = self.batch_norm(x)
        x = self.linear(x)
        return x

    def forward_submodels(self, x: Dict[str, Tensor]):
        x_snp, x_image = x[0], x[1]
        snp_out =  [self.submodel_snp[modality](x_snp[modality]) for modality in self.modalities_snp]
        image_out = [self.submodel_image[modality](x_image[modality]) for modality in self.modalities_image]
        return snp_out + image_out

    def modality_scores(self, p):
        return dict(zip(self.modalities, self.fusion.scores(p)))
