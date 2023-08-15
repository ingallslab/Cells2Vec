import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from networks import *



def load_data(dataset_path, split_idx, num_val, verbose=True, manual_indices=None):
    """
    Load and preprocess a padded dataset for training and validation.

    Parameters:
    - dataset_path (str): Path to the dataset file.
    - split_idx (int): Index at which to split the training_data into training and validation sets.
    - num_val (int): Number of classes to remove from the training data for final validation, to check model robustness and generalizing ability.
    - verbose (bool, optional): If True, print information about the dataset and processing steps.
    - new_ver (bool, optional): If True, use the new version of data processing. Default is False.
    - manual_indices (list, optional): List of manually selected indices for the final validation set. 
      If provided, num_val is ignored. Default is None.

    Returns:
    - train_set (Tensor): Processed training dataset.
    - val_set (Tensor): Processed validation dataset.
    - final_boss (Tensor): Validation data that was dropped from the training set.
    - mean (float): Mean value of the original dataset.
    - std (float): Standard deviation of the original dataset.
    """
 
    dataset = torch.load(dataset_path)
    
    if verbose:
        print(f"Dataset Shape: {dataset.shape}")
        print(f"Dataset Standard Deviation: {dataset.std()}, Dataset Mean: {dataset.mean()}")
    
    if num_val >= len(dataset):
        raise ValueError("n should be less than the length of the array")
    
    if manual_indices is None:
        selected_indices = torch.randint(high=len(dataset), size=(num_val,), dtype=torch.long).tolist()
    else:
        selected_indices = manual_indices
    
    to_drop = dataset[selected_indices]
    
    to_drop_indices = [i for i in range(len(dataset)) if i not in selected_indices]
    drop_free = dataset[to_drop_indices]
    
    end_idx = drop_free.shape[1]
    
    if verbose:
        print(f"Raw Training Set Shape: {drop_free.shape}, Final Validation Set Shape: {to_drop.shape}")
        print(f"Selected Validation Indices: {selected_indices}")
    
    train_set = drop_free[:, :split_idx, :, :].clone()
    train_set = train_set.permute(0, 1, 3, 2)
    val_set = drop_free[:, split_idx:end_idx, :, :].clone()
    val_set = val_set.permute(0, 1, 3, 2)
    final_boss = to_drop.permute(0, 1, 3, 2)
    
    if verbose:
        print(f"Train Set Shape: {train_set.shape}, Val Set Shape: {val_set.shape}\n")
    
    mean = dataset.mean()
    std = dataset.std()
    
    return train_set, val_set, final_boss, mean, std


def triplet_loss_sampler(data):
    """
    Sample a single triplet of anchor, positive, and negative samples for triplet loss.

    Parameters:
    - data (Tensor): The input data tensor of shape (num_classes, num_channels,seq_length).

    Returns:
    - anchor_sample (Tensor): The anchor sample tensor.
    - positive_sample (Tensor): The positive sample tensor.
    - negative_sample (Tensor): The negative sample tensor.
    - class_label (int): The class label for the anchor sample.  
    """
    num_classes = data.shape[0]
    sim_range = data.shape[1]

    # Select random class label
    class_label = torch.randint(1, num_classes + 1, (1,))

    # Select random samples from the anchor class
    anchor_class_idx = class_label

    negative_class_idx = torch.randint(1, num_classes + 1, (1,))
    while negative_class_idx == anchor_class_idx:
        negative_class_idx = torch.randint(1, num_classes + 1, (1,))

    anchor_sample_idx = torch.randint(0, sim_range, (1,))
    positive_sample_idx = torch.randint(0, sim_range, (1,))
    while positive_sample_idx == anchor_sample_idx:
        positive_sample_idx = torch.randint(0, sim_range, (1,))
    negative_sample_idx = torch.randint(0, sim_range, (1,))

    anchor_sample_class = data[anchor_class_idx - 1].squeeze()
    positive_sample_class = data[anchor_class_idx - 1].squeeze()
    negative_sample_class = data[negative_class_idx - 1].squeeze()

    anchor_sample = anchor_sample_class[anchor_sample_idx].squeeze()
    positive_sample = positive_sample_class[positive_sample_idx].squeeze()
    negative_sample = negative_sample_class[negative_sample_idx].squeeze()
    return anchor_sample, positive_sample, negative_sample, class_label.item()


def dataset_generator(num_iterations, data):
    """
    Generate a dataset of triplets iteratively for training using the provided data.

    Parameters:
    - num_iterations (int): Number of iterations to generate triplets, same as size of final training dataset.
    - data (Tensor): The input data tensor of shape (num_classes, sim_range, ...).

    Returns:
    - anc_list (Tensor): List of anchor samples.
    - pos_list (Tensor): List of positive samples.
    - neg_list (Tensor): List of negative samples.
    - lab_list (Tensor): List of class labels for anchor samples.  
    """

    anc_list, pos_list, neg_list, lab_list = [], [], [], []
    for _ in range(num_iterations):
        anc, pos, neg, lab = triplet_loss_sampler(data)
        anc_list.append(anc)
        pos_list.append(pos)
        neg_list.append(neg)
        lab_list.append(lab)
    anc_list = torch.stack(anc_list)
    pos_list = torch.stack(pos_list)
    neg_list = torch.stack(neg_list)
    lab_list = torch.tensor(lab_list)
    return anc_list, pos_list, neg_list, lab_list


def unravel(tensor):
    """
    Remove padding from the end of a tensor.

    Parameters:
    - tensor (Tensor): The input tensor.

    Returns:
    - unraveled_tensor (Tensor): The tensor with consecutive zero rows removed from the end.
    """
    consecutive_zeros = 0
    tensor = tensor.transpose(1, 0)
    for i, row in enumerate(tensor):
        if all(elem == 0 for elem in row):
            consecutive_zeros += 1
            if consecutive_zeros == 4:
                return tensor[:i].transpose(1, 0)
        else:
            consecutive_zeros = 0

    return tensor.transpose(1, 0)


class QuadrupletDataset(Dataset):

    """
    Initialize a quadruplet dataset for triplet loss training.

    Parameters:
    - tensor1 (Tensor): Anchor samples.
    - tensor2 (Tensor): Positive samples.
    - tensor3 (Tensor): Negative samples.
    - tensor4 (Tensor): Class labels.
    - unravel (bool, optional): If True, unravel consecutive zero rows in input tensors. Default is False.
    """

    def __init__(self, tensor1, tensor2, tensor3, tensor4, unravel=False):
        assert len(tensor1) == len(tensor2) == len(tensor3) == len(
            tensor4), "All tensors must have the same length."
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.tensor3 = tensor3
        self.tensor4 = tensor4
        self.unravel = unravel

    def __len__(self):
        return len(self.tensor1)

    def __getitem__(self, idx):
        if self.unravel:
            tensor1 = unravel(self.tensor1[idx])
            tensor2 = unravel(self.tensor2[idx])
            tensor3 = unravel(self.tensor3[idx])
        else:
            tensor1 = self.tensor1[idx]
            tensor2 = self.tensor2[idx]
            tensor3 = self.tensor3[idx]

        return tensor1, tensor2, tensor3, self.tensor4[idx]


def train_test_val_generator(train_data, test_data, val_data, num_samples, unravel=False):
    """
    Generate training, testing, and validation datasets for triplet loss training.

    Parameters:
    - train_data (Tensor): Training data tensor.
    - test_data (Tensor): Testing data tensor.
    - val_data (Tensor): Validation data tensor.
    - num_samples (int): Total number of samples in the dataset.
    - unravel (bool, optional): If True, unravel consecutive zero rows in input tensors. Default is False.

    Returns:
    - train_dataset (QuadrupletDataset): Training dataset.
    - test_dataset (QuadrupletDataset): Testing dataset.
    - val_dataset (QuadrupletDataset): Validation dataset.
    """

    train_anc, train_pos, train_neg, train_lab = dataset_generator(
        round(0.80*num_samples), train_data)
    test_anc, test_pos, test_neg, test_lab = dataset_generator(300, test_data)
    val_anc, val_pos, val_neg, val_lab = dataset_generator(
        round(0.30*num_samples), val_data)
    if unravel:
        train_dataset = QuadrupletDataset(
            train_anc, train_pos, train_neg, train_lab, unravel=True)
        test_dataset = QuadrupletDataset(
            test_anc, test_pos, test_neg, test_lab, unravel=True)
       
        val_dataset = QuadrupletDataset(
            val_anc, val_pos, val_neg, val_lab, unravel=True)
    else:
        train_dataset = QuadrupletDataset(
            train_anc, train_pos, train_neg, train_lab)
        test_dataset = QuadrupletDataset(
            test_anc, test_pos, test_neg, test_lab)
        val_dataset = QuadrupletDataset(val_anc, val_pos, val_neg, val_lab)
    return train_dataset, test_dataset, val_dataset


def load_model_from_checkpoint(checkpoint_path):
    """
    Load a trained instance of an encoder from a checkpoint with a Triplet instance of the encoder.

    Parameters:
    - checkpoint_path (str): Path to the checkpoint file.

    Returns:
    - model (CausalCNNEncoder): Loaded pre-trained model.
    """
    checkpoint = torch.load(checkpoint_path)

   
    config = checkpoint['config']

    model = CausalCNNEncoder(
        in_channels=config['Input Size'],
        channels=config['channels'],
        depth=config['Depth'],
        reduced_size=config['Reduced Size'],
        out_channels=config['output_size'],
        kernel_size=config['kernel_size']
    )

    
    dikt = checkpoint['model_state_dict']
    keys = dikt.keys()
    net_dikt = model.state_dict()
    desired_keys = list(net_dikt.keys())
    new_state_dict = {key.replace("embeddingnet.", ""): dikt[key] for key in keys if any(
        desired_key in key for desired_key in desired_keys)}
    model.load_state_dict(new_state_dict)

    model.to('cuda')

    return model


def make_numpy_array(pickle_to_dict):

    columns_to_exclude = [0, 3, 4, 5, 7, 8, 10, 12, 17]

    property_dict = {'time': [], 'id': [], 'parent': [], 'label': [],
                     'cellType': [], 'divideFlag': [], 'cellAge': [], 'growthRate': [], 'LifeHistory': [],
                     'startVol': [], 'targetVol': [], 'pos': [], 'radius': [], 'length': [], 'dir': [],
                     'ends0': [], 'ends1': [], 'strainRate': [], 'strainRate_rolling': []}

    for key in pickle_to_dict['cellStates'].keys():
        cell_state = pickle_to_dict['cellStates'][key]
        property_dict['time'].append(cell_state.time)
        property_dict['id'].append(cell_state.id)
        property_dict['label'].append(cell_state.label)
        property_dict['cellType'].append(cell_state.cellType)
        property_dict['divideFlag'].append(cell_state.divideFlag)
        property_dict['cellAge'].append(cell_state.cellAge)
        property_dict['growthRate'].append(cell_state.growthRate)

        if hasattr(cell_state, 'LifeHistory'):
            property_dict['LifeHistory'].append(cell_state.LifeHistory)
        else:
            property_dict['LifeHistory'].append(0)

        property_dict['startVol'].append(cell_state.startVol)
        property_dict['targetVol'].append(cell_state.targetVol)
        property_dict['pos'].append(
            np.sqrt(np.sum(np.power(cell_state.pos, 2))))
        property_dict['radius'].append(cell_state.radius)
        property_dict['length'].append(cell_state.length)
        property_dict['dir'].append(np.arctan2(
            cell_state.dir[1], cell_state.dir[0]))
        property_dict['ends0'].append(
            np.sqrt(np.sum(np.power(cell_state.ends[0], 2))))
        property_dict['ends1'].append(
            np.sqrt(np.sum(np.power(cell_state.ends[1], 2))))
        property_dict['strainRate'].append(cell_state.strainRate)
        property_dict['strainRate_rolling'].append(
            cell_state.strainRate_rolling)

    for bac_id in property_dict['id']:
        if bac_id in pickle_to_dict['lineage']:
            property_dict['parent'].append(pickle_to_dict['lineage'][bac_id])
        else:
            property_dict['parent'].append(0)

    df_bacteria = pd.DataFrame.from_dict(property_dict)

    df_bacteria.fillna(0, inplace=True)

 
    df_bacteria = df_bacteria.astype(float)
    tensor = torch.tensor(df_bacteria.values)
    mask = torch.ones(tensor.shape[1], dtype=torch.bool)
    mask[columns_to_exclude] = False


    tensor_filtered = tensor[:, mask]

    return tensor_filtered
