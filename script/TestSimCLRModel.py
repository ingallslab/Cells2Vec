from torchvision.models import resnet50, ResNet50_Weights
from script.SimCLR.TestSimCLR import test_simclr


def resnet_base_model():
    # ResNet50 used as default base model(encoder)
    # weights='ResNet50_Weights.DEFAULT'
    base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # call the test_simclr function
    test_simclr(img_dir, image_size, base_model, temperature, num_workers, output_dir)


def custom_model():

    # base model
    base_model = resnet50(weights=None)
    # if you want to use a custom model
    model_parameters = '../saved_models/model.pt'

    # call the test_simclr function
    test_simclr(img_dir, image_size, base_model, temperature, num_workers, output_dir, model_parameters,
                custom_model=True)


if __name__ == '__main__':
    # set the path to the image directory
    img_dir = "../input/"

    # image size after resizing
    image_size = 96

    # set the temperature
    temperature = 0.07

    # num of cpu core
    num_workers = 4

    # output directory to save representations
    output_dir = '../data'

    # resnet model
    resnet_base_model()
    # custom model
    # custom_model()
