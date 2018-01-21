from median_filter import MedianFilter
from linear_model import LinearModel
from image_threshold import ImageThreshold


def linear_regression(train_path):
    test_dir = "./test"

    lm = LinearModel(alpha=0.1)
    lm.train(train_path=train_path, show_plot=True)
    lm.run(test_dir=test_dir)


def median_filtering(train_name):
    mf = MedianFilter(file_path="./test/" + train_name)
    mf.find_background(kernel_size=5)

    mf.run(save_to_file=True, dir="./test_cleaned/median/", name=train_name)


def thresholder(train_name):
    image_thresholder = ImageThreshold(file_path="./test/" + train_name)
    image_thresholder.run(block_size=31, save_to_file=True,
                          dir="./test_cleaned/threshold/", name=train_name)

lst = ["4.png", "178.png", "181.png", "145.png"]

linear_regression("6.png")
