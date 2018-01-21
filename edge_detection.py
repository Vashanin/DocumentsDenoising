import numpy as np
import loader
import matplotlib.pyplot as plt
from sklearn import metrics
import


class EdgeDetection:
    def __init__(self, file_path):
        image_vector, width, height = loader.image_to_vector(file_path)

        self.image_vector = image_vector
        self.image_matrix = image_vector.reshape((width, height))
        self.width = width
        self.height = height

    def run(self, save_to_file=False, **kwargs):
        result = cv3.Canny(self.image_matrix)

        if save_to_file:
            loader.save_to_file(kwargs["dir"], kwargs["name"], result.reshape((self.width, self.height)))

        return result

def main():
    file_path = "./train/131.png"

    mf = EdgeDetection(file_path=file_path)
    mf.run(save_to_file=True, dir="./131", name="edges.png")

main()
