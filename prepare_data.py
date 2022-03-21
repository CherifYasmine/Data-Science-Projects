import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from utils import split_data, order_test_set


if __name__ == "__main__":
    if False:
        path_to_data = "C:\\Users\\hp\Downloads\\archive\\Train"
        path_to_save_train = "C:\\Users\\hp\Downloads\\archive\\training_data\\train"
        path_to_save_val = "C:\\Users\\hp\Downloads\\archive\\training_data\\val"
        split_data(path_to_data, path_to_save_train, path_to_save_val)
    path_to_images = "C:\\Users\\hp\Downloads\\archive\\Test"
    path_to_csv = "C:\\Users\\hp\Downloads\\archive\\Test.csv"
    order_test_set(path_to_images, path_to_csv)