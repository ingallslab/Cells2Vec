import glob
from script.SimCLR.Transform import SimCLRData_train_Transform, SimCLRData_test_Transform
from PIL import Image
from random import shuffle


def find_images_list(img_path):
    # finding existing images in a specific directory with jpg format
    images_path_list = list(glob.glob(img_path + '/*.jpg'))
    return images_path_list


def read_images(images_path_list, size=96, status='train'):

    """
    @param images_path_list str images directory
    @param size float  image size after resizing
    @param status str 'train' or 'test' During testing, only image resizing is performed. However,
    during training, two transformations, namely random_resized_crop and color_jitter, are applied to the images.
    @return images list of tensors
    """

    if status == 'train':
        # Preprocessing configuration
        data_transforms = SimCLRData_train_Transform(size=size)
    else:
        # Preprocessing configuration
        data_transforms = SimCLRData_test_Transform(size=size)

    # Read images
    images = []
    for path in images_path_list:
        original_img = Image.open(path)
        img = data_transforms(original_img)
        # show images
        # plot_transforms(original_img, img)
        images.append(img)
    return images


def train_test_split(images_path_list, test_size=0.3, num_iterate=1):

    # Shuffle should be repeated `num_iterate` times
    for i in range(num_iterate):
        shuffle(images_path_list)

    train_images = images_path_list[:round(len(images_path_list) * (1-test_size))]
    test_images = images_path_list[round(len(images_path_list) * (1 - test_size)):]

    return train_images, test_images
