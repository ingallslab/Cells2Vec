from script.Siamese.Siamese import SiameseNetwork
import torch
from script.SimCLR.SimCLR import SimCLRModel
from script.SimCLR.WorkingWithDataset import read_images, find_images_list
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


def train_siamese_network(test_images_path, image_size, base_model, batch_size, num_epochs, temperature, num_workers,
                          output_dir, model_parameters=None, custom_model=False):

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
    # create the Siamese Network
    siamese_network = SiameseNetwork()
    if custom_model:
        # Load the saved parameters
        simclr_model.load_state_dict(torch.load(model_parameters))

    # send model to device
    simclr_model.to(device)
    siamese_network.to(device)

    images_path_list = find_images_list(test_images_path)
    # read images
    test_images_path = read_images(images_path_list, size=image_size, status='test')
    # load data
    data_loader = DataLoader(test_images_path, shuffle=False, num_workers=num_workers, pin_memory=True)

    representation_list = []
    # Set the model to evaluation mode
    simclr_model.eval()
    with torch.no_grad():
        for i, x_i in enumerate(data_loader):
            # Pass the inputs through the model and calculate the loss
            representation = simclr_model(x_i)
            representation_list.append(representation)

    print(len(representation_list))
    # create a TensorDataset from the list of tensors
    dataset = TensorDataset(*representation_list)
    # create a DataLoader from the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # train the network
    for epoch in range(num_epochs):
        for i, x_i in enumerate(data_loader):
            # Pass the inputs through the model and calculate the loss
            distance = siamese_network(x_i)
            print(distance)





