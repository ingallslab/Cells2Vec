import torch.nn.functional as F
import os
import torch
import numpy as np
import pandas as pd
import gzip
from torch.nn.utils.rnn import pack_sequence

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
directory = 'PATH'

folders = [folder for folder in os.listdir(
    directory) if os.path.isdir(os.path.join(directory, folder))]

final_tensors = []
arranged_tensors = []

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

    tensored_stuff = torch.cat(tensor_list, dim=0)
    arranged_tensors.append(tensored_stuff)

#Sort Simulations based on sequence length (total number of timesteps)
lengths=[]
sorted_list=sorted(arranged_tensors, key=lambda x: x.shape[0], reverse=True)
#Sort Simulations based on sequence length (total number of timesteps)
lengths=[]
for i in range(335):
    lengths.append(sorted_list[i].shape[0])

#Exclude garbage columns using below magic

columns_to_exclude = [0, 3, 4, 5, 7, 8, 10, 12, 17]

# Get the indices of the columns to keep
columns_to_keep = [i for i in range(
   arranged_tensors.size(2)) if i not in columns_to_exclude]

# Exclude the specified columns
filtered_input = arranged_tensors[:, :, columns_to_keep]
#[ 335,1592,10]

padded_input=torch.nn.utils.rnn.pad_sequence(filtered_input, batch_first=True)
# [335,1592,19]


packed_input=torch.nn.utils.rnn.pack_padded_sequence(filtered_input, batch_first=True).float()
# Output is a packedSequence Object


