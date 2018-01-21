import numpy as np
from sklearn import cluster, metrics
import loader
from skimage import filters


class ImageThreshold:
    def __init__(self, file_path):
        image_vector, width, height = loader.image_to_vector(file_path)

        self.image_vector = image_vector
        self.image_matrix = image_vector.reshape((width, height))
        self.width = width
        self.height = height

    def run(self, block_size=9, save_to_file=False, **kwargs):
        adaptive_thresh = filters.threshold_local(self.image_matrix, block_size=block_size)

        binary_adaptive = np.ones((self.width, self.height))
        for i in range(self.width):
            for j in range(self.height):
                if self.image_matrix[i][j] < adaptive_thresh[i][j]:
                    binary_adaptive[i][j] = self.image_matrix[i][j]
                else:
                    binary_adaptive[i][j] = 1

        if save_to_file:
            loader.save_to_file(kwargs["dir"], kwargs["name"], binary_adaptive.reshape((self.width, self.height)))

        return binary_adaptive

    def get_rmse(self, prediction):
        return metrics.mean_squared_error(self.image_matrix, prediction)

