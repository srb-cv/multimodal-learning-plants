import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

class SNPModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(10, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, output_dim, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.adaptive_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1)
        )
        self.bn_final = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = x.view(int(x.size(0)), -1)
        x = self.bn_final(x)
        return x


if __name__== '__main__':
    from datasets.snp_dataset import SNPDataset

    dataset_csv = '/data/varshneya/clean_data_di/traits_csv/begin_of_flowering/dataPreprocess/begin_of_flowering_snp_image_unadjusted.csv'
    bins = SNPDataset.CHROMOSOME_BINS
    data_module = SNPDataset(dataset_csv, bins)

    X,y = data_module[0]
    print(X)
    print(y)
    model = SNPModel(output_dim=128)
    pred = model(X['A1'].unsqueeze_(0))
    print(pred)