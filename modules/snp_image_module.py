from datasets.flowering_dataset import FloweringDataset
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.argparse import from_argparse_args
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, R2Score
#from datasets.snp_image_dataset import SNPImageDataset
from models.snp_model_bins import SNPModel 

from models.fusion_snp_image import FusionSNPIMageModel
from typing import List
import csv
import matplotlib.pyplot as plt
from torchvision.models import resnet18



class SNPImageModule(pl.LightningModule):
    def __init__(self,
                 image_modalities: List[str],
                 snp_modalities: List[str],
                 latent_dim_image: int = 32,
                 latent_dim_snp: int = 4,
                 reg_param: float = 0.01,
                 p: float = 1.,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model(image_modalities, snp_modalities, latent_dim_image, latent_dim_snp)
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
            'r2 score/validation': R2Score()
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
        return loss #+ self._fusion_regularizer()

    def training_epoch_end(self, outputs):
        pass
        # self._log_modality_scores()

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
        #print(loss)        
        return {"mse":loss, "mae": self.test_mae_metric(model_out, y)}

    def test_epoch_end(self, outputs) -> None:
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        arg_group = parent_parser.add_argument_group(cls.__name__)
        arg_group.add_argument('--latent-dim-image', type=int, default=32,
                               help="Dimentionality of latent space for image modality"
                                    "i.e. dimensionality of submodels' output (default: 32)")
        arg_group.add_argument('--latent-dim-snp', type=int, default=4,
                               help="Dimentionality of latent space for snp modality"
                                    "i.e. dimensionality of submodels' output (default: 4)")
        arg_group.add_argument('--reg-param', type=float, default=0.01,
                               help="Fusion regularization parameter (default: 0.01)")
        arg_group.add_argument('--p', type=float, default=1.,
                               help="Value of 'p' for fusion block regularization")
        arg_group.add_argument('--learning-rate', type=float, default=1e-3,
                               help="Learning rate (default: 1e-3)")

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    def _build_model(self, image_modalities: List[str], snp_modalities, latent_dim_image: int, latent_dim_snp: int):
        snp_submodels = {}
        image_submodels = {}
        for modality in snp_modalities:
            model = SNPModel(latent_dim_snp)
            snp_submodels[modality] = model
        
        for modality in image_modalities:
            model = resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, latent_dim_image)
            image_submodels[modality] = model
        return FusionSNPIMageModel(submodels_image=image_submodels,
        submodels_snp=snp_submodels,
        latent_dim_image=latent_dim_image,
        latent_dim_snp=latent_dim_snp,
         out_dim=1)

    def _fusion_regularizer(self):
        return self.reg_param * self.model.fusion.regularizer(self.p)

    def _log_modality_scores(self):
        log_dict = {f"modality_scores/{modality}": score
                    for modality, score in self.model.modality_scores(self.p).items()}
        self.log_dict(log_dict)


if __name__== '__main__':
    from datamodules.snp_image_datamodule import SNPImageDatamodule
    from datasets.snp_image_dataset import SNPImageDataset

    dataset_csv = "/data/varshneya/clean_data_di/traits_csv/begin_of_flowering/combined/BeginOfFlowering_Clean_non_adjusted_image_snp_BIN_combined.csv"
    data_root = "/data/varshneya/clean_data_di"
    wave_lens=['0nm','530nm']
    bins=['bin_0','bin_1']
    datamodule = SNPImageDatamodule(dataset_csv, data_root,bins=bins,
                    wave_lens=wave_lens, batch_size=2)
    datamodule.setup()
    module = SNPImageModule(image_modalities=wave_lens,snp_modalities=bins)
    loader = datamodule.train_dataloader()
    X,y = next(iter(loader))
    pred = module(X)
    print(pred)