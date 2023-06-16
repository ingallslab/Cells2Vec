import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import  DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

#parser = argparse.ArgumentParser(description='LSTM Autoencoder Training')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data = '/SLURM_TMPDIR/data/compressed_input.pt.gz'
in_tensor=torch.load('/home/drajwade/scratch/yes/clean_padded_data.pt')
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
train_ratio = 0.8
test_ratio = 0.2
#val_ratio = 0.1
dataset=array.float()
#dataset=dataset.to(device)

# Calculate the sizes of each split
train_size = int(train_ratio * len(dataset))
test_size = int(test_ratio * len(dataset))
#val_size = len(dataset) - train_size - test_size

# Split the dataset
train_set, test_set = random_split(
    dataset, [train_size, test_size])

"""
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):

        super(LSTMAutoencoder, self).__init__()

        self.lstm_encoder = nn.LSTM(input_size, hidden_size,
                                    num_layers, dropout=dropout_rate, batch_first=True)

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
"""
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, l1_regularization=0.3, l2_regularization=0.0):
        super(LSTMAutoencoder, self).__init__()

        self.lstm_encoder = nn.LSTM(input_size, hidden_size,
                                    num_layers, dropout=dropout_rate, batch_first=True)

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
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization

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
        decoded_output, _ = self.lstm_decoder(encoded_output)
        decoded_output = decoded_output.permute(0, 2, 1)

        final_output1 = self.conv_de_block1(decoded_output)
        final_output2 = self.conv_de_block2(final_output1)
        final_output3 = self.conv_de_block3(final_output2)
        final_output4 = self.conv_de_block4(final_output3)

        return final_output4

    def l1_regularization_loss(self) -> torch.Tensor:
        l1_loss = torch.tensor(0.0).to(device)
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return l1_loss

    def l2_regularization_loss(self) -> torch.Tensor:
        l2_loss = torch.tensor(0.0).to(device)
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return l2_loss

    def compute_loss(self, outputs, inputs):
        loss_fn = nn.MSELoss()
        reconstruction_loss = loss_fn(outputs, inputs)

        l1_loss = self.l1_regularization_loss()
        l2_loss = self.l2_regularization_loss()

        total_loss = reconstruction_loss + self.l1_regularization * l1_loss + self.l2_regularization * l2_loss

        return total_loss


def train(model, train_set, test_set, batch_size, criterion, optimizer, scheduler, num_epochs, device, save_path, resume_training):
    best_loss = float('inf')
    early_stop_counter = 0
    show_graph = 0
    start_epoch = 0  # Initialize the starting epoch

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    # Check if a saved model exists and load it
    if os.path.isfile(save_path):
        if resume_training is True:
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            print(f"Resuming training from epoch {start_epoch}")

    writer = SummaryWriter(f'/scratch/drajwade/yes/runs/v4.6/hidden_size:{hidden_size}learning_rate:{learning_rate}scheduler:{scheduler_type}dropout_rate:{dropout_rate}')  # Create TensorBoard writer

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0

        model.train()  # Set model to training mode

        train_progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_progress_bar:
            batch = batch.to(device)  # Move batch to GPU if available

            reconstructed_output = model(batch)
            loss = criterion(reconstructed_output.permute(0, 2, 1), batch)
            loss = model.compute_loss(reconstructed_output.permute(0, 2, 1), batch)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if show_graph == 0:
                writer.add_graph(model, batch)
                show_graph = 1

            # train_progress_bar.set_postfix({"Train Loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_loss:.4f}")

        scheduler.step()  # Update learning rate based on scheduler

        # Log learning rate in TensorBoard
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        # Evaluate on validation set
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)  # Move batch to GPU if available

                reconstructed_output = model(batch)

                #loss = criterion(reconstructed_output.permute(0, 2, 1), batch)
                loss = model.compute_loss(reconstructed_output.permute(0, 2, 1), batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}")

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
                'best_loss': best_loss,
                'args': args
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

    writer.close()  # Close TensorBoard writer

    return model


# Create the argument parser
parser = argparse.ArgumentParser(description='LSTM Autoencoder Training')

# Add arguments for different configurations
parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for LSTM and Convolutional blocks')
parser.add_argument('--input_size', type=int, default=256, help='Input size for the LSTM Autoencoder')
parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the LSTM Autoencoder')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for LSTM Autoencoder')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change in validation loss for early stopping')
parser.add_argument('--scheduler_type', type=str, choices=['step', 'multi_step', 'exponential', 'cosine', 'reduce_lr_on_plateau', 'cyclic', 'one_cycle'],
                    default='reduce_lr_on_plateau', help='Type of scheduler to use')
parser.add_argument('--resume_training', type=bool, default=False, help='Set True to resume training from a saved model')
parser.add_argument('--batch_size', type=int, default=32, help='Set Batch Size as per how rich thou art')
args = parser.parse_args()

hidden_size = args.hidden_size
input_size = args.input_size
num_layers = args.num_layers
num_epochs = args.num_epochs
learning_rate = args.learning_rate
dropout_rate = args.dropout_rate
patience = args.patience
min_delta = args.min_delta
scheduler_type = args.scheduler_type
resume_training=args.resume_training
batch_size=args.batch_size


model = LSTMAutoencoder(input_size, hidden_size, num_layers, dropout_rate)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
#criterion=LSTMAutoencoder.compute_loss()



# Create the scheduler based on the scheduler_type
if scheduler_type == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
elif scheduler_type == 'multi_step':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
elif scheduler_type == 'exponential':
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
elif scheduler_type == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
elif scheduler_type == 'reduce_lr_on_plateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
elif scheduler_type == 'cyclic':
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, mode='triangular', cycle_momentum=False)
elif scheduler_type == 'one_cycle':
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=num_epochs, epochs=num_epochs, steps_per_epoch=train_size)

# Call the train function with the created objects and arguments
trained_model = train(model, train_set, test_set, batch_size, criterion, optimizer, scheduler, num_epochs, device, f'/scratch/drajwade/yes/checkpoints/v6_dim{hidden_size}b{batch_size}lr{learning_rate}scheduler{scheduler_type}.pt', resume_training)


