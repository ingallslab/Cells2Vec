import torch
from script.SimCLR.SimCLR import SimCLRModel
from script.SimCLR.WorkingWithDataset import read_images, find_images_list
from torch.utils.data import DataLoader
import pandas as pd


def test_simclr(test_images_path, image_size, base_model, temperature, num_workers, output_dir, model_parameters=None,
                custom_model=False):

    """
    goal: testing SimCLR model
    @param test_images_path str training images directory
    @param image_size float image size after resizing
    @param base_model basic model for SimCLR
    @param temperature float used in NT-Xent loss function (it should be positive)
    """

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('your device is: ' + str(device))
    print("Number of workers:", num_workers)

    # create the SimCLR model
    simclr_model = SimCLRModel(base_encoder=base_model, status='test')
    if custom_model:
        # Load the saved parameters
        simclr_model.load_state_dict(torch.load(model_parameters))

    # send model to device
    simclr_model.to(device)

    images_path_list = find_images_list(test_images_path)
    # read images
    test_images_path = read_images(images_path_list, size=image_size, status='test')
    # load data
    data_loader = DataLoader(test_images_path, shuffle=False, num_workers=num_workers, pin_memory=True)

    representation_dict = {}
    # Set the model to evaluation mode
    simclr_model.eval()
    with torch.no_grad():
        for i, x_i in enumerate(data_loader):
            # Pass the inputs through the model and calculate the loss
            representation = simclr_model(x_i)
            representation_dict[images_path_list[i].split('\\')[-1].split('.jpg')[0]] = representation.tolist()[0]

    # make a dataframe
    df = pd.DataFrame(representation_dict)
    df.to_csv(output_dir + '/representation.csv', index=False)

