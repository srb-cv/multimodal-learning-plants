from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
from pytorch_lightning.core.mixins import hparams_mixin
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules.snp_datamodule import SNPDataModule
from modules.snp_module import SNPModule
from datamodules.base_datamodule import DataModule
from utils import int_or_str_type

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

DATASET_FILE = "/data/varshneya/clean_data_di/traits_csv/adaptation_ger/AdaptationGer_Clean_mapped_BINS_non-adjusted.csv"


def main(args):
    logger = TensorBoardLogger(save_dir=args.logdir, name=args.name, version=args.version, default_hp_metric=False)
    datamodule = SNPDataModule.from_argparse_args(args)
    print(datamodule.bins)
    module = SNPModule.from_argparse_args(args, modalities=datamodule.bins)
    module.hparams.update({'data': datamodule.hparams})
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[EarlyStopping(monitor="loss/validation", patience=15)])
    #trainer.fit(model=module, datamodule=datamodule)
    test_pretrained_model(trainer, datamodule, module)

def test_pretrained_model(trainer, datamodule, module):
    model = SNPModule.load_from_checkpoint(
        checkpoint_path = 'logs/adaptation_ger_BINS_non-adjusted/version_1/checkpoints/epoch=12-step=1767.ckpt',
        hparams_file = 'logs/adaptation_ger_BINS_non-adjusted/version_1/hparams.yaml',
        modalities=datamodule.bins
    )
    print(datamodule.bins)
    trainer.test(model=model, datamodule= datamodule)
 

if __name__ == '__main__':
    parser = ArgumentParser()
    # Add logging args
    group = parser.add_argument_group('Logging')
    group.add_argument('--logdir', type=str, default='./logs',
                       help="Directory where logs will be stored (default: ./logs)")
    group.add_argument('--name', type=str, default='',
                       help="Experiment name. If specified, it is used as the experiment-specific subdirectory name")
    group.add_argument('--version', type=int_or_str_type,
                       help="Experiment version. It is used as the run-specific subdirectory name. "
                            "If integer is given, 'version_${VERSION}' format is used. If version is not specified, "
                            "the log directory will be inspected for existing versions and the next available version "
                            "will automatically be assigned")
    # Add datamodule args
    SNPDataModule.add_argparse_args(parser)
    # Add module args
    SNPModule.add_model_specific_args(parser)
    # Add trainer args
    pl.Trainer.add_argparse_args(parser)
    # Add args from base datamodule
    #DataModule.add_argparse_args(parser)
    # Parse all arguments
    args = parser.parse_args()
    main(args)
