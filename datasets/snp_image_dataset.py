import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch


class SNPImageDataset(Dataset):
    #ALL_WAVE_LENS = ['0nm', '530nm', '570nm', '670nm', '700nm', '730nm', '780nm']
    ALL_WAVE_LENS = ['0nm']
    #ALL_BINS = ['bin_'+str(i) for i in range(101)]
    ALL_BINS = ['A1', 'A10', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    IMG_EXT = '.tif'
    vocabulary = ['A', 'C', 'G', 'K', 'M', 'R', 'S', 'T', 'W', 'Y']
    encoder = OneHotEncoder(categories=[vocabulary], handle_unknown='ignore')

    def __init__(self,
                 dataset_csv: str,
                 data_root: str,
                 wave_lens: List[str],
                 bins: List[str],
                 transform=None,
                 year_split: str=None):
        wave_lens_set = set(wave_lens)
        if len(wave_lens) != len(wave_lens_set):
            raise ValueError('Specified wave lengths must be unique.')
        if not wave_lens_set.issubset(self.ALL_WAVE_LENS):
            unknown_wave_lens = list(wave_lens_set - set(self.ALL_WAVE_LENS))
            raise ValueError(f"Modalities not known: {unknown_wave_lens}. Available modalities: {self.ALL_WAVE_LENS}")
        self.dataset_csv = dataset_csv
        self.data_root = data_root
        self.wave_lens = wave_lens
        self.bins = bins
        self.transform = transform
        self.data = self.make_dataset_df(self.dataset_csv, self.wave_lens, self.bins, year_split)


    def __getitem__(self, idx):
        snps, images, target = self._get_sample(idx)
        if self.transform is not None:
            images, target = self.transform(images, target)
        snp_dict = {key: torch.from_numpy(self._one_hot_encode(value)).float() for key, value in snps.items()}
        return (snp_dict, images), target

    def __len__(self):
        return len(self.data)

    def _one_hot_encode(self, sequence):
        seq_arr = np.array(list(sequence)).reshape(-1, 1)
        return SNPImageDataset.encoder.fit_transform(seq_arr).toarray().T

    def _get_sample(self, idx):
        row = self.data.iloc[idx]
        split_plot_code = row['plotCode'].split('_')
        target = row['observation'].astype(np.float32)
        year = split_plot_code[2]
        location = split_plot_code[1]
        data_dir = os.path.join(self.data_root, "season"+str(year), "DeepIntegrate_Images_"+location+"_"+str(year))
        images = {wave_len: pil_loader(os.path.join(data_dir, row[wave_len].upper() + self.IMG_EXT))
                  for wave_len in self.wave_lens}
        snps = {bin: row[bin] for  bin in self.bins}
        return snps, images, target

    def preview(self, idx):
        images, target = self._get_sample(idx)
        print(f"Days to flowering: {target}")
        for wave_len, image in images.items():
            plt.figure()
            plt.title(wave_len)
            plt.imshow(image)
        plt.show()

    @staticmethod
    def make_dataset_df(dataset_csv, wave_lens, bins, year_split):
        df = pd.read_csv(dataset_csv)
        # filter = (df['processingStatus'] == 'uncropped') & \
        #          (df['observation'] > 10)
        # df = df[filter].drop(columns=['trait', 'processingStatus'])
        # df = df[df['waveLength'].isin(wave_lens)]
        # df = df[df['waveLength'].isin(FloweringDataset.ALL_WAVE_LENS)]
        trait = df.loc[0,'trait']
        print(f"Testing for the trait: {trait}")
        diff_wavelens = list(set(SNPImageDataset.ALL_WAVE_LENS) - set(wave_lens))
        diff_bins = list(set(SNPImageDataset.ALL_BINS) - set(bins))
        df = df.drop(columns=diff_wavelens+diff_bins)
        if 'begin of flowering' in trait:
            print(f"Calculating number of days for the trait {trait}")
            df['observation'] = df['observation'].astype(int) + 1  # adding one to make day of the year
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['daysToFlowering'] = df['observation'] - df['date'].dt.day_of_year
            df = df.drop(columns=['observation'])
            df.rename(columns={'daysToFlowering':'observation'}, inplace=True)
            df = df[df['observation'].isin(range(-5,5))]
        else:
            print("Normalizing the observation values with min-max scaling")
            df['observation'] = df['observation'].astype(np.float32)
            df.loc[:,'observation'] = (df['observation'] - df['observation'].min()) / (df['observation'].max() - df['observation'].min())
        df = df.dropna(subset=wave_lens+bins).reset_index()
        df = df.filter(['plotCode','date','observation','harvestYear','locationNumber']+wave_lens+bins)
        df = df.astype({'harvestYear': str, 'locationNumber': str})
        if year_split is None:
            return df
        flag, loc, year = year_split.split('_')    
        loc = '1' if loc=='HOH' else '2'
        if flag == 'val':
            return df[(df['harvestYear']==str(year)) & (df['locationNumber']==loc)]
        elif flag == 'train':
            return df[~((df['harvestYear']==str(year)) & (df['locationNumber']==loc))]
            
           


if __name__== '__main__':
    dataset_csv = "/data/varshneya/clean_data_di/traits_csv/begin_of_flowering/BeginOfFlowering_Clean_non-adjusted_mapped_chromosome_images.csv"
    data_root = "/data/varshneya/clean_data_di"
    bins = SNPImageDataset.ALL_BINS
    wave_lens = SNPImageDataset.ALL_WAVE_LENS
    dataset = SNPImageDataset(dataset_csv, data_root, wave_lens, bins)
    X,y = dataset[0]
    print(X)
    print(y)
