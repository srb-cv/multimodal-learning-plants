from datasets.flowering_dataset import FloweringDataset
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.argparse import from_argparse_args
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, R2Score, PearsonCorrcoef
from torchvision.models import resnet18

from models.fusion_model import FusionModel
from typing import List
import csv
import matplotlib.pyplot as plt

class FloweringModule(pl.LightningModule):
    def __init__(self,
                 modalities: List[str],
                 latent_dim: int = 512,
                 reg_param: float = 0.01,
                 p: float = 1.,
                 learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore='modalities')
        self.model = self._build_model(modalities, latent_dim)
        self.reg_param = reg_param
        self.p = p
        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()
        self.train_metrics = MetricCollection({
            'mean squared error/train': MeanSquaredError(),
            'mean absolute error/train': MeanAbsoluteError(),
            'r2 score/train': R2Score()
        })
        self.val_metrics = MetricCollection({
            'mean squared error/validation': MeanSquaredError(),
            'mean absolute error/validation': MeanAbsoluteError(),
            'r2 score/validation': R2Score(),
            'rscore/validation':PearsonCorrcoef()
        })
        self.test_mae_metric = MeanAbsoluteError()

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {'loss/train': float("nan"),
                                                   'loss/validation': float("nan")})

    def training_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        loss = self.loss(model_out, y.type(torch.float32))
        self.log("loss/train", loss)
        self.log_dict(self.train_metrics(model_out, y))
        return loss + self._fusion_regularizer()

    def training_epoch_end(self, outputs):
        self._log_modality_scores()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        loss = self.loss(model_out, y.type(torch.float32))
        self.log("loss/validation", loss)
        self.log_dict(self.val_metrics(model_out, y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        loss = self.loss(model_out, y.type(torch.float32))
        self.log("loss/validation", loss)
        self.log_dict(self.val_metrics(model_out, y))
        
        return {"mse":loss, "mae": self.test_mae_metric(model_out, y)}

    def test_epoch_end(self, outputs) -> None:
        out_tensor = torch.stack([x["mae"] for x in outputs])
        mean_mse = torch.mean(out_tensor)
        max_mae = torch.max(out_tensor)
        min_mae = torch.min(out_tensor)
        median_mae = torch.median(out_tensor) 
        q_25_50_75 = torch.quantile(out_tensor, torch.tensor([0.25, 0.5, 0.75]).to('cuda:0'))
        sorted_indices = torch.argsort(out_tensor)
        print(f'Mean MAE: {mean_mse}')
        print(f'Max MAE: {max_mae}')
        print(f'Min MAE: {min_mae}')
        print(f'Median MAE: {median_mae}')
        print(f'Obtained Quantile Scores:{q_25_50_75}')
        print(f'Indices with smallest 5 mae: {sorted_indices[:5]}')
        print(f'Indices with largest 5 mae: {sorted_indices[-5:]}')
        #torch.save(out_tensor, "out_rgb_tensor.pt")
        log_dict = {f"{modality}": score.detach().cpu().item()
                    for modality, score in self.model.modality_scores(self.p).items()}
        print(log_dict)
        #exit(0)
        with open('csv/begin_of_flowering_wavelengths_weights.csv','w') as f:
            w = csv.writer(f)
            w.writerows(log_dict.items())

        plt.bar(range(len(log_dict)), log_dict.values(), align='center')
        plt.xticks(range(len(log_dict)), list(log_dict.keys()))
        plt.savefig('csv/begin_of_flowering_wavelengths.png')


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        arg_group = parent_parser.add_argument_group(cls.__name__)
        arg_group.add_argument('--latent-dim', type=int, default=512,
                               help="Dimentionality of latent space, "
                                    "i.e. dimensionality of submodels' output (default: 512)")
        arg_group.add_argument('--reg-param', type=float, default=0.01,
                               help="Fusion regularization parameter (default: 0.01)")
        arg_group.add_argument('--p', type=float, default=1.,
                               help="Value of 'p' for fusion block regularization")
        arg_group.add_argument('--learning-rate', type=float, default=1e-4,
                               help="Learning rate (default: 1e-4)")

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    def _build_model(self, modalities: List[str], latent_dim: int):
        submodels = dict()
        for modality in modalities:
            model = resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, latent_dim)
            submodels[modality] = model
        return FusionModel(submodels=submodels, latent_dim=latent_dim, out_dim=1)

    def _fusion_regularizer(self):
        return self.reg_param * self.model.fusion.regularizer(self.p)

    def _log_modality_scores(self):
        log_dict = {f"modality_scores/{modality}": score
                    for modality, score in self.model.modality_scores(self.p).items()}
        self.log_dict(log_dict)
