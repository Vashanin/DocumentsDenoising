import numpy as np
from sklearn import cluster
import loader
from skimage import filters


class ImageThreshold:
    def __init__(self, file_path):
        image_vector, width, height = loader.image_to_vector(file_path)

        self.image_vector = image_vector
        self.image_matrix = image_vector.reshape((width, height))
        self.width = width
        self.height = height

    def run(self, save_to_file=False, **kwargs):
        kmeans = cluster.KMeans(n_clusters=3)

        clustered_points = kmeans.fit_predict(self.image_vector.reshape((-1, 1)))
        cluster_centres = kmeans.cluster_centers_.reshape(3,)

        separated_by_clusters = {0: [], 1: [], 2: []}
        for i in range(len(clustered_points)):
            separated_by_clusters[clustered_points[i]].append(self.image_vector[i])

        print(cluster_centres)

        clusters = {cluster_centres[0]: 0, cluster_centres[1]: 1, cluster_centres[2]: 2}
        sorted_cluster_centres = np.sort(cluster_centres)

        lower_threshold = (max(separated_by_clusters[clusters[sorted_cluster_centres[0]]])
                           + min(separated_by_clusters[clusters[sorted_cluster_centres[1]]])) / 2

        upper_threshold = (max(separated_by_clusters[clusters[sorted_cluster_centres[1]]])
                           + min(separated_by_clusters[clusters[sorted_cluster_centres[2]]])) / 2

        cleaned_image = np.ones((self.width, self.height))

        for i in range(self.width):
            for j in range(self.height):
                if upper_threshold <= self.image_matrix[i][j]:
                    cleaned_image[i][j] = 1
                else:
                    cleaned_image[i][j] = 0

        if save_to_file:
            loader.save_to_file(kwargs["dir"], kwargs["name"], cleaned_image.reshape((self.width, self.height)))

        return cleaned_image

    def cheat(self, **kwargs):
        adaptive_thresh = filters.threshold_local(self.image_matrix, 31)
        binary_adaptive = self.image_matrix > adaptive_thresh

        print(binary_adaptive)

        loader.save_to_file(kwargs["dir"], kwargs["name"], binary_adaptive.reshape((self.width, self.height)))


def main():
    image_thresholder = ImageThreshold(file_path="./test/214.png")
    # image_thresholder.run(save_to_file=True, dir="./1", name="threshold.png")
    image_thresholder.cheat(dir="./214", name="threshold.png")

main()
