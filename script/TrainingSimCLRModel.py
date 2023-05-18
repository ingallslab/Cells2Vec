from torchvision.models import resnet50
from script.SimCLR.TrainSimCLR import train_simclr

if __name__ == '__main__':
    # set the path to the image directory
    img_dir = "../input/"

    # image size after resizing
    image_size = 96
    # batch size
    batch_size = 4

    # set the number of epochs
    num_epochs = 10

    # set the temperature
    temperature = 0.07

    # num of cpu core
    num_workers = 4

    # if model needs to be saved
    save_model = True
    output_dir = '../saved_models'

    # ResNet50 used as default base model(encoder)
    # weights='ResNet50_Weights.DEFAULT'
    base_model = resnet50(weights=None)

    # call the train_simclr function
    train_simclr(img_dir, image_size, base_model, batch_size, num_epochs, temperature, num_workers, save_model,
                 output_dir)
