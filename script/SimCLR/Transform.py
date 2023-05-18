from torchvision import transforms
import matplotlib.pyplot as plt


def plot_transforms(original_img, img):
    plt.imshow(original_img)
    plt.show()

    for img_transformed in img:
        # Convert the tensor image to a numpy array
        transformed_image = img_transformed.permute(1, 2, 0).detach().numpy()

        # Display the transformed image
        plt.imshow(transformed_image)
        plt.show()


# Define the data transforms for SimCLR
def SimCLRData_train_Transform(size=96, brightness=0.8, contrast=0.2, saturation=0.2, hue=0.2):
    random_resized_crop = transforms.Compose([
            # Randomly crop a section of the image and resize it to 96x96 pixels
            transforms.RandomResizedCrop(size=size),
            # Convert the image to a PyTorch tensor
            transforms.ToTensor(),
            # I found this configuration by:  ResNet50_Weights.DEFAULT.transforms()
            # configuration:
            # ImageClassification( crop_size=[224], resize_size = [232], mean = [0.485, 0.456, 0.406],
            # std = [0.229, 0.224, 0.225], interpolation = InterpolationMode.BILINEAR)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    color_jitter = transforms.Compose([
        # Randomly crop a section of the image and resize it to 96x96 pixels
        transforms.Resize(size=size),
        #  Randomly adjust image color values
        transforms.RandomApply([
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)]
            , p=0.8),
        # Convert the image to a PyTorch tensor
        transforms.ToTensor(),
        # I found this configuration by:  ResNet50_Weights.DEFAULT.transforms()
        # configuration:
        # ImageClassification( crop_size=[224], resize_size = [232], mean = [0.485, 0.456, 0.406],
        # std = [0.229, 0.224, 0.225], interpolation = InterpolationMode.BILINEAR)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def transform(image):
        # Apply the transforms to the image and return a tuple of two transformed images
        return random_resized_crop(image), color_jitter(image)

    return transform


# Define the data transforms for SimCLR
def SimCLRData_test_Transform(size=96, brightness=0.8, contrast=0.2, saturation=0.2, hue=0.2):
    random_resized = transforms.Compose([
            # Randomly crop a section of the image and resize it to 96x96 pixels
            transforms.Resize(size=size),
            # Convert the image to a PyTorch tensor
            transforms.ToTensor(),
            # I found this configuration by:  ResNet50_Weights.DEFAULT.transforms()
            # configuration:
            # ImageClassification( crop_size=[224], resize_size = [232], mean = [0.485, 0.456, 0.406],
            # std = [0.229, 0.224, 0.225], interpolation = InterpolationMode.BILINEAR)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def transform(image):
        # Apply the transforms to the image and return a transformed image
        return random_resized(image)

    return transform
