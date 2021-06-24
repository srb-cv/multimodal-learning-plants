from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules.flowering_datamodule import FloweringDatamodule
from modules.flowering_module import FloweringModule
from utils import int_or_str_type

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# DATASET_FILE = '/home/balinski/deep-integrate/data/processed/image-traits/HOH_2019_20/processedDFbeginofflowering.csv'
# DATA_ROOT = '/data/varshneya/DeepIntegrate_Images_HOH_2020/DeepIntegrate_Images_HOH_2020'


def main(args):
    logger = TensorBoardLogger(save_dir=args.logdir, name=args.name, version=args.version, default_hp_metric=False)
    datamodule = FloweringDatamodule.from_argparse_args(args)
    module = FloweringModule.from_argparse_args(args, modalities=datamodule.wave_lens)
    module.hparams.update({'data': datamodule.hparams})
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model=module, datamodule=datamodule)


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
    FloweringDatamodule.add_argparse_args(parser)
    # Add module args
    FloweringModule.add_model_specific_args(parser)
    # Add trainer args
    pl.Trainer.add_argparse_args(parser)
    # Parse all arguments
    args = parser.parse_args()
    main(args)
