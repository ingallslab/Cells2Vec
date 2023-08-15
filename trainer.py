from data_utils import *
from networks import *
from losses import *
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import time


def train_model(args,train_loader, val_loader, model, criterion, optimizer, num_epochs, device, tb_log_dir, checkpoint_dir, config):
    
    os.makedirs(checkpoint_dir, exist_ok=True) 
    writer = SummaryWriter(tb_log_dir)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    checkpoint_path = ''

    for epoch in range(num_epochs):
      
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()

        for _, i in enumerate(train_loader):
            anchor, positive, negative, _ = i
         
            
            if args.use_cnn:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            elif args.use_lstm or args.use_gru:
                anchor, positive, negative = anchor.permute(0, 2, 1).to(device), positive.permute(0, 2, 1).to(device), negative.permute(0, 2, 1).to(device)
             
            optimizer.zero_grad()
            anchor_embed, pos_embed, neg_embed = model(anchor, positive, negative)
            loss = criterion(anchor_embed, pos_embed, neg_embed)
        
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss_avg = train_loss_sum / len(train_loader)
        train_losses.append(train_loss_avg)
  
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for _, i in enumerate(val_loader):
                anchor, positive, negative, _ = i
                
                if args.use_cnn:
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                elif args.use_lstm or args.use_gru:
                    anchor, positive, negative = anchor.permute(0, 2, 1).to(device), positive.permute(0, 2, 1).to(device), negative.permute(0, 2, 1).to(device)

                anchor_embed, pos_embed, neg_embed = model(anchor, positive, negative)
                loss = criterion(anchor_embed, pos_embed, neg_embed)

                val_loss_sum += loss.item()

        val_loss_avg = val_loss_sum / len(val_loader)
        val_losses.append(val_loss_avg)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'config': config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss_avg:.4f} - Validation Loss: {val_loss_avg:.4f} - Time taken: {time.time() - start_time:.2f} seconds")
        writer.add_scalar('Loss/train', train_loss_avg, epoch)
        writer.add_scalar('Loss/validation', val_loss_avg, epoch)

    writer.close()
    return train_losses, val_losses, checkpoint_path


