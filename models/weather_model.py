import torch
import torch.nn as nn
import torch.nn.functional as F

class Weather_Model(nn.Module):
    def __init__(self, device, inp_dim:int = 6, hid_dim1:int = 128, hid_dim2: int = 50, hid_dim3: int = 6, out_dim: int = 6, 
    num_layers: int = 2, bias: bool = False, batch_first : bool = True, LSTM_layers: int = 2, hid_dim_per_layer: list = [50,30]):
        super(Weather_Model, self).__init__()
        self.device = device
        # self.inp_dim = inp_dim
        # self.hid_dim1 = hid_dim1
        # self.hid_dim2 = hid_dim2
        # self.hid_dim3 = hid_dim3
        # self.out_dim = out_dim
        self.num_layers = num_layers

        self.LSTM_layers = LSTM_layers
        self.hid_dim_per_layer = hid_dim_per_layer
        self.lstm_layer_list = []
        self.h = []
        self.c = []
        
        # self.lstm = nn.LSTM(input_size = self.inp_dim, hidden_size = self.hid_dim1, num_layers = self.num_layers)
        # self.lstm2 = nn.LSTM(input_size = self.hid_dim1, hidden_size = self.hid_dim2, num_layers = self.num_layers)
        # self.l1 = nn.Linear(in_features = self.hid_dim2, out_features = self.hid_dim3)

        for i in range(self.LSTM_layers):
            self.lstm_layer_list.append(nn.LSTM(input_size = self.hid_dim_per_layer[i], hidden_size = self.hid_dim_per_layer[i+1], num_layers = self.num_layers).to(self.device))
        self.l1 = nn.Linear(in_features = self.hid_dim_per_layer[i+1], out_features = self.hid_dim_per_layer[i+2])

    def forward(self, x):
        # h_0 = torch.zeros(self.num_layers, x.size(1), self.hid_dim1).to(self.device) #hidden state
        # c_0 = torch.zeros(self.num_layers, x.size(1), self.hid_dim1).to(self.device) #cell state
        # h_2 = torch.zeros(self.num_layers, x.size(1), self.hid_dim2).to(self.device) #hidden state
        # c_2 = torch.zeros(self.num_layers, x.size(1), self.hid_dim2).to(self.device) #cell state
        # print("input shape", x.shape)  #[32, 1, 6]
        # out, (h, c) = self.lstm(x, (h_0, c_0)) # out shape- [32, 1, 50], c.shape- [1, 1, 50], h.shape- [1, 1, 50]
        # print("lstm 1 out {}, h {}, c {} shape".format(out.shape, h.shape, c.shape))
        # out2, (h2, c2) = self.lstm2(out, (h_2, c_2)) # out shape- [32, 1, 30], c.shape- [1, 1, 30], h.shape- [1, 1, 30]
        # print("lstm 2 out {}, h {}, c {} shape".format(out2.shape, h2.shape, c2.shape))
        # lout = self.l1(out2) #[32, 1, 6]
        # print("l1 output shape", lout.shape)
        # print("out2 shape", out2.shape)
        # return lout, (h2,c2)

        out = x
        for i in range(self.LSTM_layers):
            self.h.append(torch.zeros(self.num_layers, x.size(1), self.hid_dim_per_layer[i+1]).to(self.device)) #hidden state
            self.c.append(torch.zeros(self.num_layers, x.size(1), self.hid_dim_per_layer[i+1]).to(self.device)) #cell state
            out, (self.h[i], self.c[i]) = self.lstm_layer_list[i](out, (self.h[i], self.c[i])) # out shape- [32, 1, 50], c.shape- [1, 1, 50], h.shape- [1, 1, 50]
            # print("lstm {} out {}, h {}, c {} shape".format(i, out.shape, self.h[i].shape, self.c[i].shape))
        
        # print("out shape - ", out.shape) #[32, 1, 30]
        lout = self.l1(out) #[32, 1, 6]
        # print("l1 output shape", lout.shape, self.h[i].shape, self.c[i].shape) #torch.Size([32, 1, 6]) torch.Size([1, 1, 30]) torch.Size([1, 1, 30])

        return lout, (self.h[i],self.c[i])
