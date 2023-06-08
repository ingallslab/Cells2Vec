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

#data = '/SLURM_TMPDIR/data/compressed_input.pt.gz'
data = '/home/drajwade/scratch/yes/compressed_input.pt.gz'
with gzip.open(data, 'rb') as f:
    array1 = torch.load(f)
reshaped_tensor = array1.view(-1, 19)
#scaler = MinMaxScaler()
scaler=StandardScaler()
scaler.fit(reshaped_tensor)
scaled_tensor = scaler.transform(reshaped_tensor)
array = torch.from_numpy(scaled_tensor)
array = array.view(335, 6144, 19)
print(f"Normalize array shape:{array.shape}")
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1
dataset=array.float()
dataset=dataset.to(device)

# Calculate the sizes of each split
train_size = int(train_ratio * len(dataset))
test_size = int(test_ratio * len(dataset))
val_size = len(dataset) - train_size - test_size

# Split the dataset
train_set, test_set, val_set = random_split(
    dataset, [train_size, test_size, val_size])

# Create data loaders for each split
batch_size = 32  # Set the desired batch size
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Print the sizes of each split
print(f"Train set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")
print(f"Validation set size: {len(val_set)}")

 #Move model to GPU if available

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer, save_path):
    best_loss = float('inf')
    early_stop_counter = 0
    show_graph = 0
    

    start_epoch = 0  # Initialize the starting epoch

    # Check if a saved model exists and load it
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0

        model.train()  # Set model to training mode

        train_progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_progress_bar:
            batch = batch.to(device)  # Move batch to GPU if available

            reconstructed_output = model(batch)

            loss = criterion(reconstructed_output.permute(0, 2, 1), batch)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if show_graph==0:
                writer.add_graph(model, batch)
                show_graph=1

            #train_progress_bar.set_postfix({"Train Loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)  # Update learning rate based on average loss

        # Evaluate on validation set
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)  # Move batch to GPU if available

                reconstructed_output = model(batch)

                loss = criterion(reconstructed_output.permute(0, 2, 1), batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}")

        # Log metrics in TensorBoard
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)

        # Save model if early stopping condition is met
        if avg_val_loss < best_loss - min_delta:
            best_loss = avg_val_loss
            early_stop_counter = 0

            # Save the model state and other relevant information
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }

            if torch.cuda.device_count() > 1:
                checkpoint['model_state_dict'] = model.module.state_dict()

            torch.save(checkpoint, save_path)
            print("Model saved")

        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model



class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):

        super(LSTMAutoencoder, self).__init__()

        self.lstm_encoder = nn.LSTM(input_size, hidden_size,
                                    num_layers, dropout=0.2, batch_first=True)

        self.conv_en_block1 = nn.Sequential(
            nn.Conv1d(in_channels=19, out_channels=32,
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
            nn.Conv1d(in_channels=32, out_channels=19,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=19),
            nn.ReLU()
        )

        self.lstm_decoder = nn.LSTM(hidden_size, input_size,
                                    num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.permute(0, 2, 1)
        conv_input1 = self.conv_en_block1(x)
        conv_input1 = self.dropout(conv_input1)
        conv_input2 = self.conv_en_block2(conv_input1)
        conv_input2 = self.dropout(conv_input2)
        conv_input3 = self.conv_en_block3(conv_input2)
        conv_input3 = self.dropout(conv_input3)
        conv_input4 = self.conv_en_block4(conv_input3)
        conv_input4 = self.dropout(conv_input4)
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
hidden_size = 16
num_layers = 4
num_epochs = 100
learning_rate = 0.001
dropout_rate = 0.2
patience = 20
min_delta = 0.001
lr_factor = 0.5
lr_patience = 3
lr_min = 1e-6

writer = SummaryWriter(f'/scratch/drajwade/yes/runs/v4.2/hidden_size:{hidden_size}')
model = LSTMAutoencoder(input_size, hidden_size, num_layers, dropout_rate)
model = model.to(device)  # Move model to GPU if available
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float('inf')
early_stop_counter = 0

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=lr_factor, patience=lr_patience, min_lr=lr_min)

trained_model = train(model, train_loader, val_loader, criterion,
                      optimizer, scheduler, num_epochs, device, writer, save_path=f'/scratch/drajwade/yes/checkpoints/v4_dim{hidden_size}.pt')

