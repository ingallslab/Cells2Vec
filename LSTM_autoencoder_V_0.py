import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import gzip
from sklearn.preprocessing import MinMaxScaler

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data = '/SLURM_TMPDIR/data/compressed_input.pt.gz'
data='/scratch/drajwade/yes/compressed_input.pt.gz'
with gzip.open(data, 'rb') as f:
    array1 = torch.load(f)
print(f"Raw array shape:{array1.shape}")
reshaped_tensor= array1.view(-1,19)
scaler=MinMaxScaler()
scaler.fit(reshaped_tensor)
scaled_tensor=scaler.transform(reshaped_tensor)
array=torch.from_numpy(scaled_tensor)
array=array.view(335,6144,19)
print(f"Normalize array shape:{array.shape}")
array=array.to(device)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size,
                               num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size,
                               num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        encoded_output, _ = self.encoder(x)
        encoded_output = self.dropout(encoded_output)
        decoded_output, _ = self.decoder(encoded_output)
        return decoded_output


input_size = 19
hidden_size = 256
num_layers = 4
batch_size = 32
num_epochs = 1000
learning_rate = 0.0001
dropout_rate = 0.2
patience = 20
min_delta = 0.001
lr_factor = 0.5
lr_patience = 3
lr_min = 1e-6

model = LSTMAutoencoder(input_size, hidden_size, num_layers, dropout_rate)
model = model.to(device)  # Move model to GPU if available

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for parallel processing")
    model = nn.DataParallel(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_data = array.float()

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)

best_loss = float('inf')
early_stop_counter = 0

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=lr_factor, patience=lr_patience, min_lr=lr_min)

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)  # Move batch to GPU if available

        reconstructed_output = model(batch)

        loss = criterion(reconstructed_output, batch)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    scheduler.step(avg_loss)  # Update learning rate based on average loss

    if avg_loss < best_loss - min_delta:
        best_loss = avg_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Save the model state
save_path = '/scratch/drajwade/yes/v0_01.pt'
if torch.cuda.device_count() > 1:
    torch.save(model.module.state_dict(), save_path)
else:
    torch.save(model.state_dict(), save_path)
