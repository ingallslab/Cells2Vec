from torch.optim import SGD
from torch.optim import lr_scheduler
import torch
from script.SimCLR.SimCLR import SimCLRModel
from script.SimCLR.WorkingWithDataset import read_images, find_images_list
from torch.utils.data import DataLoader


def train_simclr(training_images_path, image_size, base_model, batch_size, num_epochs, temperature, num_workers,
                 save_model=False, output_dir=None):

    """
    goal: training SimCLR model
    @param training_images_path str training images directory
    @param image_size float image size after resizing
    @param base_model basic model for SimCLR
    @param batch_size int
    @param num_epochs int
    @param temperature float used in NT-Xent loss function (it should be positive)
    """

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('your device is: ' + str(device))
    print("Number of workers:", num_workers)

    # create the SimCLR model
    simclr_model = SimCLRModel(base_encoder=base_model, temperature=temperature)
    # send model to device
    simclr_model.to(device)

    images_path_list = find_images_list(training_images_path)
    # read images
    training_images_path = read_images(images_path_list, size=image_size, status='train')
    # load data
    data_loader = DataLoader(training_images_path, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=True)

    # Define optimizer and learning rate scheduler
    # learning rate: default value = 0.48 (0.3 * batch_size / 256)
    learning_rate = 0.48
    # LARS optimizer
    # LARS (Layer-wise Adaptive Rate Scaling) is an optimization algorithm that is designed
    # to adjust the learning rate of each weight in a neural network layer-wise,
    # based on the local gradient of the loss function. The idea is to scale the learning rate of each weight
    # by a factor that is proportional to the ratio of the L2 norm of the weights and the L2 norm of the gradients.
    optimizer = SGD(simclr_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate/50)

    # Train the model
    for epoch in range(num_epochs):
        # train() tells to the model that you are training the model.
        simclr_model.train()
        total_loss = 0
        for i, (x_i, x_j) in enumerate(data_loader):

            optimizer.zero_grad()
            # Pass the inputs through the model and calculate the loss
            loss = simclr_model((x_i, x_j))
            # Compute the gradients and update the parameters
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Update the learning rate scheduler
            scheduler.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(data_loader)))

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in simclr_model.state_dict():
        print(param_tensor, "\t", simclr_model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    if save_model:
        # Save the learned parameters of the model
        torch.save(simclr_model.state_dict(), output_dir + '/model.pt')
        # torch.save(simclr_model, output_dir + '/model.pth')

