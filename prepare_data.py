import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from utils import split_data


if __name__ == "__main__":
    path_to_data = "C:\\Users\\hp\Downloads\\archive\\Train"
    path_to_save_train = "C:\\Users\\hp\Downloads\\archive\\training_data\\train"
    path_to_save_val = "C:\\Users\\hp\Downloads\\archive\\training_data\\val"
    split_data(path_to_data, path_to_save_train, path_to_save_val)