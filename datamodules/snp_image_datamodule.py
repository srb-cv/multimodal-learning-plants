from typing import Union, Optional, List

import torch
from torch.utils.data import random_split
from torch.utils.data import Subset
from torchvision.datasets.vision import StandardTransform
from torchvision.transforms import Compose, Resize, ToTensor

from datamodules.base_datamodule import DataModule
from datasets.snp_image_dataset import SNPImageDataset
from transforms import ModalityWiseTransform
from utils import get_split_lengths, int_or_float_type


class SNPImageDatamodule(DataModule):
    dataset_cls = SNPImageDataset

    def __init__(self,
                 dataset_csv,
                 data_root,
                 wave_lens: Optional[List[str]] = None,
                 bins: Optional[List[str]] = None,
                 val_split: Union[int, float] = 0.3,
                 seed: int = 42,
                 year_split: str = None,
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
        self.data_root = data_root
        self.wave_lens = self.dataset_cls.ALL_WAVE_LENS if wave_lens is None else wave_lens
        self.bins = self.dataset_cls.ALL_BINS if bins is None else bins
        self.val_split = val_split
        self.year_split = year_split
        self.seed = seed
        self.transforms = self._make_transforms()
        self.train_set = None
        self.val_set = None

    def setup(self, stage: Optional[str] = None):
        if self.year_split:
            self.train_set = self.dataset_cls(dataset_csv=self.dataset_csv,
                                   data_root=self.data_root,
                                   wave_lens=self.wave_lens,
                                   bins=self.bins,
                                   transform=self.transforms,
                                   year_split='train_'+self.year_split)
            val_set = self.dataset_cls(dataset_csv=self.dataset_csv,
                                   data_root=self.data_root,
                                   wave_lens=self.wave_lens,
                                   bins=self.bins,
                                   transform=self.transforms,
                                   year_split='val_'+self.year_split)
        else:
            dataset = self.dataset_cls(dataset_csv=self.dataset_csv,
                                   data_root=self.data_root,
                                   wave_lens=self.wave_lens,
                                   bins=self.bins,
                                   transform=self.transforms)
            split_lengths = get_split_lengths([self.val_split], len(dataset))
            self.train_set, val_set = random_split(dataset, split_lengths,
                                               generator=torch.Generator().manual_seed(self.seed))
        self.val_set = val_set if len(val_set) else None
        print(f'Number of datapoints in the train set : {len(self.train_set)}')
        print(f'Number of datapoints in the validation set: {len(self.val_set)}')

    def train_dataloader(self):
        return self._data_loader(self.train_set, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._data_loader(self.val_set)
    
    def test_dataloader(self):
        return self._data_loader(self.val_set)

    def _make_transforms(self):
        # sourcery skip: inline-immediately-returned-variable
        transforms = StandardTransform(
            transform=ModalityWiseTransform({
                wave_len: Compose([Resize((224, 224)), ToTensor()]) for wave_len in self.wave_lens
            }),
            target_transform=None
        )
        return transforms

    @classmethod
    def _add_concrete_argparse_args(cls, arg_group):
        arg_group.add_argument('--dataset-csv', type=str, required=True,
                               help="Dataset csv file")
        arg_group.add_argument('--data-root', type=str, required=True,
                               help="Drone images root directory")
        arg_group.add_argument('--wave-lens', type=str, nargs='+',
                               help="Wavelengts to use. If not specified, all available wavelengts will be used "
                                    "(default: all wavelengths)")
        arg_group.add_argument('--bins', type=str, nargs='+',
                               help="Bins to use. If not specified, all available bins will be used "
                                    "(default: all 101 bins)")
        arg_group.add_argument('--val-split', type=int_or_float_type, default=0.3,
                               help="Fraction of data (int or float) to use for validation (default: 0.3)")
        arg_group.add_argument('--seed', type=int, default=42,
                               help="Seed used for random data splits and shuffling (default: 42)")
        arg_group.add_argument('--year-split', type=str, 
                               help="Use the data from the specified enviornment as validation set. eg: HOH_2018 for Hohenlieth location in year 2018")

if __name__== '__main__':
    dataset_csv = '/data/varshneya/clean_data_di/traits_csv/begin_of_flowering/BeginOfFlowering_Clean_non-adjusted_mapped_chromosome_images.csv'
    data_root = "/data/varshneya/clean_data_di"
    datamodule = SNPImageDatamodule(dataset_csv, data_root, wave_lens=['0nm'],  year_split='HOH_2018', bins=['A1','A2'])
    datamodule.setup()
