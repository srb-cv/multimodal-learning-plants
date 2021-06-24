import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import pandas as pd
import matplotlib.pyplot as plt


class FloweringDataset(Dataset):
    ALL_WAVE_LENS = ['0nm', '530nm', '570nm', '670nm', '700nm', '730nm', '780nm']
    IMG_EXT = '.tif'

    def __init__(self,
                 dataset_csv: str,
                 data_root: str,
                 wave_lens: list[str],
                 transform=None):
        wave_lens_set = set(wave_lens)
        if len(wave_lens) != len(wave_lens_set):
            raise ValueError(f"Specified wave lengths must be unique.")
        if not wave_lens_set.issubset(self.ALL_WAVE_LENS):
            unknown_wave_lens = list(wave_lens_set - set(self.ALL_WAVE_LENS))
            raise ValueError(f"Modalities not known: {unknown_wave_lens}. Available modalities: {self.ALL_WAVE_LENS}")
        self.dataset_csv = dataset_csv
        self.data_root = data_root
        self.wave_lens = wave_lens
        self.transform = transform
        self.data = self.make_dataset_df(self.dataset_csv, self.wave_lens)

    def __getitem__(self, idx):
        images, target = self._get_sample(idx)
        if self.transform is not None:
            images, target = self.transform(images, target)
        return images, target

    def __len__(self):
        return len(self.data)

    def _get_sample(self, idx):
        row = self.data.iloc[idx]
        target = row['daysToFlowering']
        images = {wave_len: pil_loader(os.path.join(self.data_root, row[wave_len].upper() + self.IMG_EXT))
                  for wave_len in self.wave_lens}
        return images, target

    def preview(self, idx):
        images, target = self._get_sample(idx)
        print(f"Days to flowering: {target}")
        for wave_len, image in images.items():
            plt.figure()
            plt.title(wave_len)
            plt.imshow(image)
        plt.show()

    @staticmethod
    def make_dataset_df(dataset_csv, wave_lens):
        df = pd.read_csv(dataset_csv)
        filter = (df['processingStatus'] == 'uncropped') & \
                 (df['observation'] > 10)
        df = df[filter].drop(columns=['trait', 'processingStatus'])
        df = df[df['waveLength'].isin(wave_lens)]
        df['observation'] = df['observation'].astype(int) + 1  # adding one to make day of the year
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['daysToFlowering'] = df['observation'] - df['date'].dt.day_of_year
        df = df.drop(columns=['observation'])
        df = df.pivot(index=['plotCode', 'date', 'daysToFlowering'], columns='waveLength', values='imageCode')
        df = df.dropna().reset_index()

        return df
