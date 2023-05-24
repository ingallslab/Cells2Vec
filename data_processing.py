import torch.nn.functional as F
import os
import torch
import numpy as np
import pandas as pd
import gzip


def make_numpy_array(pickle_to_dict):
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

    return df_bacteria


## First, we need the data in format {num_simulations, padded_sequence_length, num_features}

# Below is code for padding the sequences to the same length, and saving the final tensor in a compressed format

directory = 'PATH'

folders = [folder for folder in os.listdir(
    directory) if os.path.isdir(os.path.join(directory, folder))]

final_tensors = []

for folder in folders:
    folder_path = os.path.join(directory, folder)

    
    pickle_files = [file for file in os.listdir(
        folder_path) if file.endswith('.pickle')]

    
    tensor_list = []
    count_large = 0


    for pickle_file in pickle_files:
        
        pickle_to_dict = np.load(os.path.join(
            folder_path, pickle_file), allow_pickle=True)
        
        df = make_numpy_array(pickle_to_dict)

        
        tensor = torch.tensor(df.values)


        tensor_list.append(tensor)

    # Find the maximum number of observations in tensor_list
    max_num_obs = max([tensor.shape[0] for tensor in tensor_list])

    # Pad tensors with zeros to match the maximum number of observations
    padded_tensors = []
    for tensor in tensor_list:
        pad_shape = (max_num_obs - tensor.shape[0], tensor.shape[1])
        padded_tensor = F.pad(tensor, (0, 0, 0, pad_shape[0]))
        padded_tensors.append(padded_tensor)

    # Concatenate the padded tensors along the first dimension (num_observations)
    final_tensor = torch.cat(padded_tensors, dim=0)

    final_tensors.append(final_tensor)

# Pad the final tensors in final_tensors to have the same number of rows
max_num_rows = max([tensor.shape[0] for tensor in final_tensors]) 
print(sum([tensor.shape[0] for tensor in final_tensors])/len(final_tensors)) # 5054
padded_final_tensors = []
for tensor in final_tensors:
    pad_shape = (max_num_rows - tensor.shape[0], tensor.shape[1])
    padded_tensor = F.pad(tensor, (0, 0, 0, pad_shape[0]))
    padded_final_tensors.append(padded_tensor)

# Stack the final tensors along a new dimension (num_folders)
array = torch.stack(padded_final_tensors, dim=0)

# Save the final tensor array in a compressed format
output_file = 'path/file_name.pt.gz'
with gzip.open(output_file, 'wb') as f:
    torch.save(array, f)

print(array.shape)
print("Final tensor saved in compressed format.")

#To load the tensor array, use the following code:
"""
with gzip.open(output_file, 'rb') as f:
    array = torch.load(f)
print(array.shape)

"""

