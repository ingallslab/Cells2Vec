import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import gzip
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #data = '/SLURM_TMPDIR/data/compressed_input.pt.gz'
# data = '/home/drajwade/scratch/yes/experimental.pt.gz'
# with gzip.open(data, 'rb') as f:
#     array1 = torch.load(f)
# reshaped_tensor = array1.view(-1, 19)
scaler = MinMaxScaler()
#scaler=StandardScaler()
# scaler.fit(reshaped_tensor)
# scaled_tensor = scaler.transform(reshaped_tensor)
# array = torch.from_numpy(scaled_tensor)
# array = array.view(13, 28090, 19)
# print(f"Normalize array shape:{array.shape}")
# dataset=array.float()
# #dataset=TensorDataset(dataset)
# batch_size = 1 # Set the desired batch size
# val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

in_tensor=torch.load('/home/drajwade/scratch/yes/exp_clean_padded_data.pt')
num_sims=in_tensor.shape[0]
num_timesteps=in_tensor.shape[1]
num_features=in_tensor.shape[2]
reshaped_tensor=in_tensor.view(in_tensor.shape[0]*in_tensor.shape[1], in_tensor.shape[2] )
#scaler = MinMaxScaler()
scaler=StandardScaler()
scaler.fit(reshaped_tensor)
scaled_tensor = scaler.transform(reshaped_tensor)
array = torch.from_numpy(scaled_tensor)
array = array.view(num_sims, num_timesteps, num_features)
print(f"Normalized array shape:{array.shape}")


t_list=[]
for i in range(array.shape[0]):
    temp=array[i]
    #scaler.fit(temp)
    #scaled_temp=scaler.transform(temp)
    #retemp=torch.from_numpy(scaled_temp).float()
    t_list.append(temp)

#This leaves me with a list of tensors (len=13) of shape (num_timesteps,19)
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):

        super(LSTMAutoencoder, self).__init__()

        self.lstm_encoder = nn.LSTM(input_size, hidden_size,
                                    num_layers, batch_first=True)

        self.conv_en_block1 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )

        self.conv_en_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        self.conv_en_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()

        )
        self.conv_en_block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()

        )

        self.conv_de_block1 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )

        self.conv_de_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        self.conv_de_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )
        self.conv_de_block4 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=10,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=10),
            nn.ReLU()
        )

        self.lstm_decoder = nn.LSTM(hidden_size, input_size,
                                    num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.permute(0, 2, 1)
        conv_input1 = self.conv_en_block1(x)
        #conv_input1 = self.dropout(conv_input1)
        conv_input2 = self.conv_en_block2(conv_input1)
        #conv_input2 = self.dropout(conv_input2)
        conv_input3 = self.conv_en_block3(conv_input2)
        #conv_input3 = self.dropout(conv_input3)
        conv_input4 = self.conv_en_block4(conv_input3)
        #conv_input4 = self.dropout(conv_input4)
        conv_input4 = conv_input4.permute(0, 2, 1)

        encoded_output, _ = self.lstm_encoder(conv_input4)
        #encoded_output = self.dropout(encoded_output)
        decoded_output, _ = self.lstm_decoder(encoded_output)
        decoded_output = decoded_output.permute(
            0, 2, 1)             # [batch_size,256,6144]
        final_output1 = self.conv_de_block1(decoded_output)
        final_output2 = self.conv_de_block2(final_output1)
        final_output3 = self.conv_de_block3(final_output2)
        final_output4 = self.conv_de_block4(final_output3)

        return final_output4

    
input_size = 256
#hidden_size = 16
#num_layers = 4
#num_epochs = 1000
#learning_rate = 0.001
#dropout_rate = 0.2
#patience = 20
#min_delta = 0.001
#lr_factor = 0.5
#lr_patience = 3
#lr_min = 1e-6
h_dim=128
model = LSTMAutoencoder(input_size=256, hidden_size=128, num_layers=4, dropout_rate=0.2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
checkpoint = torch.load('/home/drajwade/scratch/yes/checkpoints/v5_dim128.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.to(device)

model.eval()  # Set model to evaluation mode
val_loss = 0.0

predictions=[]
with torch.no_grad():
    for i in range(13):
         # Move batch to GPU if available
        input=t_list[i]
        input=torch.unsqueeze(input, 0).float()
        input=input.to(device)
        reconstructed_output = model(input)
        #print(reconstructed_output.shape)
        reconstructed_output=reconstructed_output.permute(0,2,1)
        loss = criterion(reconstructed_output, input)
        out_cpu=reconstructed_output.cpu()
        #print(out_cpu.shape)
        predictions.append(out_cpu)
        print(f"loss: {loss.item()}")
        val_loss += loss.item()

avg_val_loss = val_loss /13
why=torch.cat(predictions)
print(f"Plz work{why.shape}")
print(f"Average Val Loss: {avg_val_loss:.4f}")
torch.save(why,f'/home/drajwade/scratch/yes/{h_dim}predictions.pt')

