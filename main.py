import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import *
from networks import *
from losses import *
from trainer import *
from visualize import *
import yaml
from eval import *

""" 
This block of code is useful to get live outputs if you use a SLURM scheduler/Compute Cluster 

import sys
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

"""

def load_configurations(config_file):
    with open(config_file, 'r') as config_stream:
        config_data = yaml.safe_load(config_stream)
    return config_data['configurations']


def main(args):
    if args.config:
            configurations = load_configurations(args.config)

            for config in configurations:
                if config['name'] == args.selected_config:
                    selected_config = config
                    break
            else:
                raise ValueError(f"Selected configuration '{args.selected_config}' not found in the YAML file")


            new_config = vars(args).copy()
            new_config.update(selected_config)

            args = argparse.Namespace(**new_config)


    print("############# DATASET INFORMATION #############\n")
    num_val=args.num_val
    train_set, val_set, final_val_set, _,_ = load_data(args.data_path, split_idx=7, num_val=num_val, verbose=True,manual_indices=args.val_indices)
    print("############# DATASET INFORMATION #############\n")
    
    num_samples = args.num_samples
    train_dataset, val_dataset, _ = train_test_val_generator(train_set, val_set, final_val_set, num_samples, unravel=True)


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    if args.use_cnn:
        model_type='cnn'
        net = CausalCNNEncoder(args.in_channels, args.channels, args.depth, args.reduced_size, args.out_channels, args.kernel_size)
        model = Tripletnet(net)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        config_todo = {
        'Input Size': args.in_channels,
        'channels': args.channels,
        'Depth': args.depth,
        'Reduced Size': args.reduced_size,
        'output_size': args.out_channels,
        'kernel_size': args.kernel_size
        }


    if args.use_lstm:
        model_type='lstm'
        net=StackedLSTMEncoder(args.in_channels, args.lstm_dim, args.num_layers, args.out_channels, args.dropout)
        model=Tripletnet(net)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        config_todo={
        'magic': args.lstm_dim
        }

    if args.use_gru:
        model_type='gru'
        net=StackedGRUEncoder(args.in_channels, args.lstm_dim, args.num_layers, args.out_channels, args.dropout)
        model=Tripletnet(net)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        config_todo={
        'magic': args.lstm_dim
        }




    if args.use_multi_gpu:
        model = nn.DataParallel(model)



    if args.loss_function == 'triplet':
        criterion = TripletLoss(margin=args.margin)
    elif args.loss_function == 'triplet_cosine':
        criterion = TripletLossCosine(margin=args.margin)
    elif args.loss_function == 'triplet_with_reg':
        criterion = TripletLossWithRegularization(margin=args.margin, regularization_weight=args.reg_weight)
    elif args.loss_function == 'mixed_triplet':
        criterion=  MixedTripletLoss(alpha=0.5, margin=args.margin, margin_cosine=1)
    else:
        raise ValueError(f"Invalid loss function: {args.loss_function}")

    if args.train:

        print("############# TRAINING PARAMETERS #############\n")
        print(f"Model: {model_type}\n")
        print(f"Loss: {args.loss_function}\n")
        print(f"Starting Training using: {num_samples} Samples\n")
        print(f"Final embedding: {args.out_channels}\n")
        print(f"Layers: {args.depth}\n")
        print(f"Reduced size: {args.reduced_size}\n")
        print("############# TRAINING PARAMETERS #############\n")


        tb_log_dir = f'{args.tb_log_dir}/{model_type}_{args.num_samples}_{args.out_channels}_{args.loss_function}'

        checkpoint_dir = f'{args.checkpoint_dir}/{model_type}_{args.num_samples}_{args.out_channels}_{args.loss_function}'

        _, _, checkpoint_pathway = train_model(args,train_loader, val_loader, model, criterion, optimizer, args.num_epochs, device, tb_log_dir, checkpoint_dir, config_todo)
      
        best_model_path=checkpoint_pathway
        print("Training Completed\n")
    
    if args.evaluate:

        model  = load_model_from_checkpoint(best_model_path)
        model = model.to(device)

        print("Evaluating On Unseen Parameter Sets\n")
        evaluate_model(model, final_val_set,  device, checkpoint_dir, visualize=True)
        print("Evaluating On Seen Parameter Sets\n")
        evaluate_model(model, val_set, device, checkpoint_dir, visualize=False)
        print("Estimating Parameters using XGBoost\n")
        xgb_regress(model, args.data_path,checkpoint_dir, device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Causal CNN Training')
    parser.add_argument('--config', type=str, default=None, help='Path to a config.yaml file')
    parser.add_argument('--selected_config', type=str, default=None, help='Name of the selected configuration')
    parser.add_argument('--data_path', type=str, default='./dataset.pt', help='Path to the dataset')
    parser.add_argument('--num_val', type=int, default=7, help='Number of samples per class for training')
    parser.add_argument('--num_samples', type=int, default=2048, help='Total Samples (0.8/0.2 Split)')
    parser.add_argument('--val_indices', nargs='*', type=int, default=None,
                        help='List of manual indices (default: None)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for the triplet loss')
    parser.add_argument('--reg_weight', type=float, default=0.001, help='Weight for the regularization term')
    parser.add_argument('--tb_log_dir', type=str, default=f'./tb_runs', help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Enable multi-GPU training')
    parser.add_argument('--train', action='store_true', help='Train a model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on test dataset')
    parser.add_argument('--loss_function', type=str, default='triplet', choices=['triplet', 'triplet_cosine', 'triplet_with_reg', 'mixed_triplet'], help='Loss function for training')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_path', help='Load a Model checkpoint')
    parser.add_argument('--use_cnn', action='store_true', help='Creates a CNN Instance')
    parser.add_argument('--use_lstm', action='store_true', help='Creates a LSTM Instance')
    parser.add_argument('--use_gru', action='store_true', help='Creates a GRU Instance')
    parser.add_argument('--in_channels', type=int, default=10, help='Number of input channels for CausalCNN')
    parser.add_argument('--channels', type=int, default=10, help='Number of channels manipulated in CausalCNN')
    parser.add_argument('--depth', type=int, default=10, help='Depth of the CausalCNN')
    parser.add_argument('--reduced_size', type=int, default=200, help='Fixed length to which the output time series of CausalCNN is reduced')
    parser.add_argument('--out_channels', type=int, default=256, help='Number of output channels for CausalCNN')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size of the applied non-residual convolutions for CausalCNN')
    parser.add_argument('--num_layers',  type=int, default=2, help= 'Number of layers in stack.')
    parser.add_argument('--lstm_dim',  type=int, default=10, help= 'Final_embedding_dim).')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout for ze layers')

    args = parser.parse_args()
    main(args)
