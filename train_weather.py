from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from datasets.weather_dataset import Data_Preprocess, weather_Data_Set
from datamodules.weather_datamodule import Data_Module
from models.weather_model import Weather_Model


class train_model():
    def __init__(self, criterion, model, device, optimiser: optim, epochs: int):
        self.criterion = criterion
        self.model = model
        self.optimiser = optimiser
        self.epochs = epochs
        self.device = device
        # self.writer = writer

    def train_func(self, path:str, train_loader: DataLoader, val_loader: DataLoader, model_save_path:str):
        for epoch in tqdm(range(self.epochs)):
            train_loss_ls = []
            val_loss_ls = []
            total_train_loss = 0.0
            total_val_loss = 0.0
            saved_h = []

            for batch_data in train_loader:
                x,y = batch_data
                x = torch.reshape(x, (x.shape[0], 1, x.shape[1])).to(self.device)  #[32, 1, 6]
                y = torch.reshape(y, (y.shape[0], 1, y.shape[1])).to(self.device)  #[32, 1, 6]

                self.optimiser.zero_grad()
                pred,hc = self.model(x)  #[32, 1, 6]
                h, c = hc[0], hc[1]
                loss = self.criterion(pred, y)
                loss.backward(retain_graph=True)
                self.optimiser.step()
                total_train_loss += loss.item()
                if epoch == self.epochs-1:
                #   print("pred - ",pred.shape) #[1,6]
                #   saved_out.append(pred)
                #   print("h - ",h.shape) # [1,1,50]
                  saved_h.append(h)

            loss = total_train_loss/len(train_loader)   
            train_loss_ls.append(loss)  
            # self.writer.add_scalar('Loss/train', loss, epoch)
            if epoch%9 == 0:
                print("Training Loss {} at epoch {}".format(loss, epoch))

            for val_batch in val_loader:
                val_x, val_y = val_batch
                val_x = torch.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1])).to(self.device)  #[32, 1, 6]
                val_y = torch.reshape(val_y, (val_y.shape[0], 1, val_y.shape[1])).to(self.device)  #[32, 1, 6]

                val_pred,_ = self.model(val_x)  #[32, 1, 6]
                val_loss = self.criterion(val_pred, val_y)
                total_val_loss += val_loss

            val_loss = total_val_loss/len(val_loader)
            val_loss_ls.append(val_loss)
            # self.writer.add_scalar('Loss/validation', val_loss, epoch)
            if epoch%9 == 0:
                print("Validation Loss at epoch {} is {}".format(val_loss.item(), epoch))
        
        torch.save(self.model.state_dict(), model_save_path)
        # temp = saved_out[len(saved_out)-1][-1]
        temp = saved_h[len(saved_h)-1][-1]
        # print(temp)
        torch.save(temp, path)
        #print(len(saved_out)-1)
        #print(saved_out[len(saved_out)-1].shape)        
        #print(saved_out[len(saved_out)-1])

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

if __name__ == '__main__':
    file_path = '/data/varshneya/WeatherData/NPZ-HOH-atomar-2019.txt'
    LSTM_layers = 1
    hid_dim_per_layer = [6,512,6]
    plot_graph = False
    val_split = 33
    load = True #False #True
    num_layers = 1 
    bias = False
    batch_first = True
    lr = 1e-3
    epochs = 500
    save_path = '/data/ml/garg/WeatherLSTM/saved_models/out.pth'
    model_save_path = '/data/ml/garg/WeatherLSTM/saved_models/model.pth'

    device = get_device()
    dp = Data_Preprocess(file_path, plot_graph)
    data = dp.read_file()

    train_ds = weather_Data_Set(data[:len(data)-val_split]) 
    val_ds = weather_Data_Set(data[len(data)-val_split:])

    dl = Data_Module(batch_size = 32, shuffle = False)
    train_dl, val_dl = dl.data_loaders(train_ds, val_ds)

    model = Weather_Model(device, num_layers, bias, batch_first, LSTM_layers, hid_dim_per_layer).to(device)
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr)

    if load:
      model.load_state_dict(torch.load(model_save_path))
      model.eval()

    tm = train_model(criterion, model, device, optimiser, epochs)
    tm.train_func(save_path, train_dl, val_dl, model_save_path)
    
    