import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import gzip


output_file = '/PATH/compressed_input.pt.gz'

with gzip.open(output_file, 'rb') as f:
    array = torch.load(f)
print(array.shape)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size,
                               num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size,
                               num_layers, batch_first=True)

    def forward(self, x):
        encoded_output, _ = self.encoder(x)
        decoded_output, _ = self.decoder(encoded_output)
        return decoded_output


input_size = 19
hidden_size = 256
num_layers = 2
batch_size = 5
num_epochs = 10
learning_rate = 0.001

model = LSTMAutoencoder(input_size, hidden_size, num_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_data = array.float()

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch in train_loader:
      
        reconstructed_output = model(batch)

        loss = criterion(reconstructed_output, batch)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
