import numpy as np
from sklearn import linear_model
import loader
import matplotlib.pyplot as plt


class LinearModel:
    def __init__(self, alpha=0.1):
        self.regression = linear_model.Ridge(alpha=alpha)

    def run(self, train_path, show_plot=False):
        train_set = loader.image_to_vector("./train/" + train_path)[0]
        cleaned_set = loader.image_to_vector("./cleaned/" + train_path)[0]

        X = []
        Y = []

        min_val = 0.05
        max_val = 0.95

        for i in range(len(cleaned_set)):
            if min_val < cleaned_set[i] < max_val and min_val < train_set[i] < max_val:
                X.append(train_set[i])
                Y.append(cleaned_set[i])

        X = np.asarray(X).reshape(-1, 1)
        Y = np.asarray(Y)

        self.regression.fit(X.reshape((-1, 1)), Y)

        if show_plot:
            plt.figure(figsize=(20, 20))

            plt.xlim((0.0, 1.0))
            plt.ylim((0.0, 1.0))

            plt.scatter(train_set, cleaned_set, s=1)
            plt.plot([0, 1], self.regression.predict([[0], [1]]), color="red")

            plt.show()

    @staticmethod
    def _filter_output(item):
        result = item

        if item < 0.0:
            result = 0.0
        if item > 1.0:
            result = 1.0

        return result

    def predict(self, test_dir):
        test_set = loader.images_from_dir(test_dir)

        for image in test_set:
            name = image["name"]
            image_vector, width, height = image["data"]

            cleaned_image = self.regression.predict(image_vector.reshape((-1, 1)))
            cleaned_image = np.asarray(list(map(lambda item: LinearModel._filter_output(item), cleaned_image)))

            loader.save_to_file("./test_cleaned/linear_model/", name, cleaned_image.reshape((width, height)))


def main():
    train_path = "6.png"
    test_dir = "./test"

    lm = LinearModel(alpha=0.1)
    lm.run(train_path=train_path)
    lm.predict(test_dir=test_dir)

main()
