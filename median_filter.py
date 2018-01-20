import numpy as np
import loader
import matplotlib.pyplot as plt
import scipy.signal

class MedianFilter:
    def __init__(self, file_path, kernel_size=17):
        self.kernel_size = kernel_size

        image_vector, width, height = loader.image_to_vector(file_path)

        self.image_vector = image_vector
        self.image_matrix = image_vector.reshape((width, height))
        self.width = width
        self.height = height

        self.background = None
        self.cleaned_image = np.ones((self.width, self.height))

    def find_background(self, save_to_file=False, **kwargs):
        self.background = scipy.signal.medfilt(self.image_matrix, self.kernel_size)

        if save_to_file:
            loader.save_to_file(kwargs["dir"], kwargs["name"], self.background.reshape((self.width, self.height)))

        return self.background

    def remove_background(self, save_to_file=False, **kwargs):
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
        print(foreground)

        if save_to_file:
            loader.save_to_file(kwargs["dir"], kwargs["name"], foreground.reshape((self.width, self.height)))

        return self.cleaned_image


def main():
    mf = MedianFilter(file_path="./train/2.png")
    mf.find_background()
    mf.remove_background(save_to_file=True, dir="./", name="lets_try.png")

main()
