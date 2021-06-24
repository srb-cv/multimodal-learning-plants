from argparse import ArgumentParser
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class DataModule(pl.LightningDataModule):
    dataset_cls: Dataset

    def __init__(self,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.hparams = {'datamodule': self.__class__.__name__,
                        'batch_size': self.batch_size}

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs):
        group = parent_parser.add_argument_group(cls.__name__)
        cls._add_concrete_argparse_args(group)
        group.add_argument('--batch-size', type=int, default=32,
                           help="Number of samples to be loaded per batch (default: 32)")
        group.add_argument('--shuffle', action='store_true',
                           help="Reshuffle the training data at every epoch")
        group.add_argument('--num-workers', type=int, default=0,
                           help="Number of data loading subprocesses. '0' means that the data will be loaded "
                                "in the main process (default: '0')")
        group.add_argument('--pin-memory', action='store_true',
                           help="Place fetched data batches in CUDA pinned memory, "
                                "enabling faster data transfer to CUDA-enabled GPUs")
        group.add_argument('--drop-last', action='store_true',
                           help="Drop the last incomplete batch if the dataset size is not divisible by the "
                                "batch size. If not specified and the size of dataset is not divisible by the "
                                "batch size, then the last batch will be smaller")
        return parent_parser

    @classmethod
    def _add_concrete_argparse_args(cls, arg_group):
        pass
