import scipy.ndimage
import scipy.misc

import os
import numpy as np


def save_to_file(dir_path, name, image_matrix):
    scipy.misc.imsave(dir_path + name, image_matrix)


def image_to_vector(image_path, normalize=True, disp=False):
    matrix_image = scipy.ndimage.imread(image_path)

    if normalize:
        matrix_image = matrix_image / 255.0

    if disp:
        print("Image: {}".format(image_path))
        print(matrix_image)

    width, height = matrix_image.shape

    return matrix_image.reshape((width * height,)), width, height


def images_from_dir(images_dir, normalize=True, disp=False):
    tree = os.walk("/home/vashanin/DataRoot/DocumentsDenoising/{}".format(images_dir))

    image_names = None

    for item in tree:
        image_names = item[2]

    result = []

    for name in image_names:
        new_image_path = "{}/{}".format(images_dir, name)
        image_vector = image_to_vector(new_image_path, normalize=normalize, disp=disp)
        result.append({"data": image_vector, "name": name})

    return result
