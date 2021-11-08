from typing import Union, Optional, List

import torch
from torch.utils.data import random_split

from datamodules.base_datamodule import DataModule
from datasets.snp_dataset import SNPDataset
from utils import get_split_lengths, int_or_float_type


class SNPDataModule(DataModule):
    dataset_cls = SNPDataset

    def __init__(self,
                 dataset_csv,
                 bins: Optional[List[str]] = None,
                 val_split: Union[int, float] = 0.2,
                 seed: int = 42,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 persistent_workers: bool = False) -> None:
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         drop_last=drop_last,
                         persistent_workers=persistent_workers)
        self.dataset_csv = dataset_csv
        self.bins = self.dataset_cls.CHROMOSOME_BINS if bins is None else bins
        self.val_split = val_split
        self.seed = seed
        self.train_set = None
        self.val_set = None

    def setup(self, stage: Optional[str] = None):
        dataset = self.dataset_cls(dataset_csv=self.dataset_csv,
                                   bins=self.bins,
                                   transform=None)
        split_lengths = get_split_lengths([self.val_split], len(dataset))
        self.train_set, val_set = random_split(dataset, split_lengths,
                                               generator=torch.Generator().manual_seed(self.seed))
        self.val_set = val_set if len(val_set) else None

    def train_dataloader(self):
        return self._data_loader(self.train_set, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._data_loader(self.val_set)
    
    def test_dataloader(self):
        return self._data_loader(self.val_set)

    @classmethod
    def _add_concrete_argparse_args(cls, arg_group):
        arg_group.add_argument('--dataset-csv', type=str, required=True,
                               help="Dataset csv file")
        arg_group.add_argument('--bins', type=str, nargs='+',
                               help="Bins to use. If not specified, all available bins will be used "
                                    "(default: all bins)")
        arg_group.add_argument('--val-split', type=int_or_float_type, default=0.2,
                               help="Fraction of data (int or float) to use for validation (default: 0.2)")
        arg_group.add_argument('--seed', type=int, default=42,
                               help="Seed used for random data splits and shuffling (default: 42)")
