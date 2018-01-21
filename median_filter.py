import numpy as np
import loader
import matplotlib.pyplot as plt
import scipy.signal
from sklearn import metrics


class MedianFilter:
    def __init__(self, file_path):
        image_vector, width, height = loader.image_to_vector(file_path)

        self.image_vector = image_vector
        self.image_matrix = image_vector.reshape((width, height))
        self.width = width
        self.height = height

        self.background = None
        self.cleaned_image = np.ones((self.width, self.height))

    def find_background(self, kernel_size=5, save_to_file=False, **kwargs):
        self.background = scipy.signal.medfilt(self.image_matrix, kernel_size)

        if save_to_file:
            loader.save_to_file(kwargs["dir"], kwargs["name"], self.background.reshape((self.width, self.height)))

        return self.background

    def run(self, save_to_file=False, **kwargs):
        if self.background is None:
            self.find_background()

        for i in range(self.width):
            for j in range(self.height):
                if self.image_matrix[i][j] < self.background[i][j]:
                    self.cleaned_image[i][j] = self.image_matrix[i][j]

        foreground = self.image_matrix - self.background
        foreground[foreground > 0] = 0

        m1 = min(foreground.reshape(self.image_vector.shape))
        m2 = max(foreground.reshape(self.image_vector.shape))

        foreground = (foreground - m1) / (m2 - m1)

        if save_to_file:
            loader.save_to_file(kwargs["dir"], kwargs["name"], foreground.reshape((self.width, self.height)))

        return self.cleaned_image

    @staticmethod
    def get_rmse(true_image, prediction):
        return metrics.mean_squared_error(true_image, prediction)
