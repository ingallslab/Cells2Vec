import numpy as np
import pandas as pd
import os
import torch
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




# Define the root directory containing the folders
root_directory = 'PATH_of_all_sims/'

# Get the list of folders in the root directory
folders = [folder for folder in os.listdir(
    root_directory) if os.path.isdir(os.path.join(root_directory, folder))]

# Initialize an empty list to store the tensor for each folder
tensor_list = []

# Process each folder
for folder in folders:
    # Define the directory for the current folder
    directory = os.path.join(root_directory, folder)

    # Get the list of pickle files in the directory
    pickle_files = [file for file in os.listdir(
        directory) if file.endswith('.pickle')]

    # Process each pickle file in the current folder
    for pickle_file in pickle_files:
        # Load the pickle file and convert it to a dictionary
        pickle_to_dict = np.load(os.path.join(
            directory, pickle_file), allow_pickle=True)
        # Assuming you have a function to convert the dictionary to a DataFrame
        df = make_numpy_array(pickle_to_dict)

        # Convert DataFrame to PyTorch tensor
        tensor = torch.tensor(df.values)

        # Append the tensor to the list
        tensor_list.append(tensor)

# Concatenate the tensors along the first dimension (num_pickles)
final_tensor = torch.cat(tensor_list, dim=0)


# Save the final tensor array in a compressed format
output_file = 'output_path/compressed_tensor.pt.gz'
with gzip.open(output_file, 'wb') as f:
    torch.save(final_tensor, f)
print(f"Final tensor saved in compressed format, with shape: {final_tensor.shape}")
