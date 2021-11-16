import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch

class SNPDataset(Dataset):

    @staticmethod
    def _get_bin_list():
        snp_ordering = "data/Position_GeneticData_ordered.txt"
        bins_df = pd.read_csv(snp_ordering,delimiter='\t')
        bins_df['bin'] = bins_df['Chromosome'] + '_' + bins_df['Position'].map(str)
        bins_group_indices_df = bins_df.groupby('bin')['Index'].apply(list)
        return list(bins_group_indices_df.index)
    
    #CHROMOSOME_BINS = _get_bin_list.__func__()
    CHROMOSOME_BINS = ['A1', 'A10', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    vocabulary = ['A', 'C', 'G', 'K', 'M', 'R', 'S', 'T', 'W', 'Y']
    encoder = OneHotEncoder(categories=[vocabulary], handle_unknown='ignore')

    def __init__(self,
                 dataset_csv: str,
                 bins: List[str],
                 transform=None,
                 year_split=None):
        bins_set = set(bins)
        if len(bins) != len(bins_set):
            raise ValueError('Specified bins must be unique.')
        if not bins_set.issubset(self.CHROMOSOME_BINS):
            unknown_bins = list(bins_set - set(self.CHROMOSOME_BINS))
            raise ValueError(f"Modalities not known: {unknown_bins}. Available modalities: {self.CHROMOSOME_BINS}")
        self.dataset_csv = dataset_csv
        self.bins = bins
        self.transform = transform
        self.data = self.make_dataset_df(self.dataset_csv, self.bins, year_split)

    def __getitem__(self, idx):
        data_dict, target = self._get_sample(idx)
        data_dict = {key: torch.from_numpy(self._one_hot_encode(value)).float() for key, value in data_dict.items()}
        if self.transform is not None:
            data_dict, target = self.transform(data_dict, target)
        return data_dict, target

    def __len__(self):
        return len(self.data)

    def _one_hot_encode(self, sequence):
        seq_arr = np.array(list(sequence)).reshape(-1, 1)
        return SNPDataset.encoder.fit_transform(seq_arr).toarray().T

    def _get_sample(self, idx):
        row = self.data.iloc[idx]
        target = row['observation']
        data_dict = {bin: row[bin] for  bin in self.bins}
        return data_dict, target

    @staticmethod
    def make_dataset_df(dataset_csv, bins, year_split):
        df = pd.read_csv(dataset_csv)
        trait = df.loc[0,'trait']
        df = df.filter(['plotCode','date','observation','harvestYear','locationNumber']+bins)
        df = df.dropna(subset=bins).reset_index()
        df = df.drop_duplicates(subset=bins+['observation'],ignore_index=True)
        if 'begin of flowering' in trait:
            print(f"Calculating number of days for the trait {trait}")
            df['observation'] = df['observation'].astype(int) + 1  # adding one to make day of the year
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['daysToFlowering'] = df['observation'] - df['date'].dt.day_of_year
            df = df.drop(columns=['observation'])
            df.rename(columns={'daysToFlowering':'observation'}, inplace=True)
            df = df[df['observation'].isin(range(-5,5))]
        df['observation'] = df['observation'].astype(np.float32)
        #df.loc[:,'observation'] = (df['observation'] - df['observation'].min()) / (df['observation'].max() - df['observation'].min())
        df = df.astype({'harvestYear': str, 'locationNumber': str})
        print(f'Number of datapoints: {len(df)}')
        if year_split is None:
            return df
        flag, loc, year = year_split.split('_')    
        loc = '1' if loc=='HOH' else '2'
        if flag == 'val':
            return df[(df['harvestYear']==str(year)) & (df['locationNumber']==loc)]
        elif flag == 'train':
            return df[~((df['harvestYear']==str(year)) & (df['locationNumber']==loc))]

if __name__== '__main__':
    dataset_csv = '/data/varshneya/clean_data_di/traits_csv/adaptation_ger/AdaptationGer_Clean_mapped_chromosome_non-adjusted.csv'
    #bins = SNPDataset.CHROMOSOME_BINS
    ALL_BINS = ['A1', 'A10', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    data_module = SNPDataset(dataset_csv, ALL_BINS)

    X,y = data_module[0]
    print(X)
    print(y)
