from median_filter import MedianFilter
from linear_model import LinearModel
from image_threshold import ImageThreshold


def linear_regression(train_path):
    test_dir = "./test"

    lm = LinearModel(alpha=0.1)
    lm.train(train_path=train_path)
    lm.run(test_dir=test_dir)


def median_filtering(train_path):
    mf = MedianFilter(file_path=train_path)
    mf.find_background(kernel_size=5)

    prediction = mf.run(save_to_file=True, dir="./131", name="median.png")

    true_image, width, height = loader.image_to_vector("./cleaned/131.png")
    print(MedianFilter.get_rmse(true_image.reshape((width, height)), prediction))


def thresholder(train_path):
    image_thresholder = ImageThreshold(file_path=train_path)
    prediction = image_thresholder.run(block_size=31, save_to_file=True, dir="./131median", name="threshold.png")

    print(image_thresholder.rmse(prediction))

