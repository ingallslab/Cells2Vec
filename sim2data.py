## CellModeller-ingallslab package necessary to read the pickle files
## Code To do that (Use terminal):

#git clone https://github.com/ingallslab/CellModeller-ingallslab.git
#cd CellModeller-ingallslab
#pip install - e . --use-pep517


import torch.nn.functional as F
import os
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import re
import csv

def make_numpy_array(pickle_to_dict):

    columns_to_exclude = [0, 3, 4, 5, 7, 8, 10, 12, 17]

    # Definition of dictionary
    property_dict = {'time': [], 'id': [], 'parent': [], 'label': [],
                     'cellType': [], 'divideFlag': [], 'cellAge': [], 'growthRate': [], 'LifeHistory': [],
                     'startVol': [], 'targetVol': [], 'pos': [], 'radius': [], 'length': [], 'dir': [],
                     'ends0': [], 'ends1': [], 'strainRate': [], 'strainRate_rolling': []}

    # Fill the dictionary
    for key in pickle_to_dict['cellStates'].keys():
        cell_state = pickle_to_dict['cellStates'][key]

        # Append values to dictionary, assign zero if attribute not present
        property_dict['time'].append(cell_state.time)
        property_dict['id'].append(cell_state.id)
        property_dict['label'].append(cell_state.label)
        property_dict['cellType'].append(cell_state.cellType)
        property_dict['divideFlag'].append(cell_state.divideFlag)
        property_dict['cellAge'].append(cell_state.cellAge)
        property_dict['growthRate'].append(cell_state.growthRate)

        # Handle 'LifeHistory' attribute
        # If not present, assign value 0
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

    # Structure of 'lineage': id : parent id
    # If no parent, assign value 0
    for bac_id in property_dict['id']:
        if bac_id in pickle_to_dict['lineage']:
            property_dict['parent'].append(pickle_to_dict['lineage'][bac_id])
        else:
            property_dict['parent'].append(0)

    # Convert dictionary to pandas DataFrame
    df_bacteria = pd.DataFrame.from_dict(property_dict)

    # Replacing NaN values with 0
    df_bacteria.fillna(0, inplace=True)

    # Convert all columns to float
    df_bacteria = df_bacteria.astype(float)
    tensor = torch.tensor(df_bacteria.values)
    mask = torch.ones(tensor.shape[1], dtype=torch.bool)
    mask[columns_to_exclude] = False

# Apply the mask to the tensor to remove the columns
    tensor_filtered = tensor[:, mask]

    return tensor_filtered


means = []
var = []
final_means = []
final_var = []

main_folder = 'PATH'
scaler = StandardScaler()


def process_subfolder(subfolder, tensor_list):
    pickle_files = sorted([f.path for f in os.scandir(
        subfolder) if f.is_file() and f.name.endswith('.pickle')])

    try:

        for pickle_file in pickle_files:
            pickle_to_dict = np.load(os.path.join(
                subfolder, pickle_file), allow_pickle=True)
            df = make_numpy_array(pickle_to_dict)
            i, j = df.shape[0], df.shape[1]
            df = scaler.fit_transform(df.view(-1, 1))
            df = torch.tensor(df).view(i, j)
            #l, m = df.shape[0], df.shape[1]
            # temp_idx = (l + 3, m)
            # temp_tensor = torch.zeros(temp_idx)
            # temp_tensor[:l, :] = df
            tensor_list.append(df)
    except RuntimeError as e:
        print(f"Exception occurred in folder: {subfolder}")
        print(f"Exception message: {e}")


def extract_values(folder_name):
    pattern = r'gamma_(\d+\.\d+)_reg_param_(\d+\.\d+)_adh_(\d+\.\d+)'
    match = re.match(pattern, folder_name)
    if match:
        gamma = float(match.group(1))
        reg_param = float(match.group(2))
        adh = float(match.group(3))
        return gamma, reg_param, adh
    else:
        return None


### Below code reads all simulations in Subfolders in a main directory
### Files are read such that for every subfolder, every ith sub-subfolder is grouped
### Example: main_folder->Iteration1, Iteration2, Iteration3....IterationN , this will return a tensor of shape (num_classes, N, padded_dim, 10)
### Also writes parameters for the simulations to a list, which can be converted to a CSV file


folders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
param_labels=[]

set_list = []
max_subfolders = max(
    [len([f.path for f in os.scandir(folder) if f.is_dir()]) for folder in folders])


for i in range(max_subfolders):

    current_set = []

    for folder in folders:

        subfolders = sorted([f.path for f in os.scandir(folder) if f.is_dir()])

        if i < len(subfolders):
            subfolder = subfolders[i]
            #print("Processing subfolder:", subfolder)
            tensor_list = []
            ##Comment out the below 3 lines if error
            values=extract_values(subfolder)
            if values:
                param_labels.append(values)
            ## End commenting out here if error    
            process_subfolder(subfolder, tensor_list)
            tensored_stuff = torch.cat(tensor_list, dim=0)

            current_set.append(tensored_stuff)

    if current_set:

        for i in range(len(current_set)):

            means.append(torch.mean(current_set[i]))
            var.append(torch.var(current_set[i]))
        final_means.append(np.mean(means))
        final_var.append(np.mean(var))
        padded_set = pad_sequence(
            current_set, batch_first=True, padding_value=0)

        set_list.append(padded_set)

max_num_cells = max([set.shape[1] for set in set_list])


padded_set_list = [
    F.pad(set, (0, 0, 0, max_num_cells - set.shape[1])) for set in set_list]


stacked_tensor = torch.stack(padded_set_list, dim=0)


print("Shape of the stacked tensor:", stacked_tensor.shape)
torch.save(stacked_tensor, 'PATH)
