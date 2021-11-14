import torch
from torch.utils.data import DataLoader, random_split
from datasets.weather_dataset import weather_Data_Set

class Data_Module():
    def __init__(self, batch_size: int = 32, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def data_loaders(self, train_data_set: weather_Data_Set, val_data_set: weather_Data_Set):
        train_dl = DataLoader(dataset = train_data_set, batch_size = self.batch_size, shuffle= self.shuffle, drop_last = True)
        val_dl = DataLoader(dataset = val_data_set, batch_size = self.batch_size, shuffle= self.shuffle, drop_last = True)
        return train_dl, val_dl




