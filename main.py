import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import split_data, order_test_set, create_generators
from models import signs_model


if __name__ == "__main__":
    # path_to_data = "C:\\Users\\hp\Downloads\\archive\\Train"
    # path_to_save_train = "C:\\Users\\hp\Downloads\\archive\\training_data\\train"
    # path_to_save_val = "C:\\Users\\hp\Downloads\\archive\\training_data\\val"
    # split_data(path_to_data, path_to_save_train, path_to_save_val)
    # path_to_images = "C:\\Users\\hp\Downloads\\archive\\Test"
    # path_to_csv = "C:\\Users\\hp\Downloads\\archive\\Test.csv"
    # order_test_set(path_to_images, path_to_csv)

    path_to_train = "C:\\Users\\hp\Downloads\\archive\\training_data\\train"
    path_to_val = "C:\\Users\\hp\Downloads\\archive\\training_data\\val"
    path_to_test = "C:\\Users\\hp\Downloads\\archive\\Test"
    batch_size = 64
    epochs = 15

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    path_to_save_model = './Models'
    chkpt_saver = ModelCheckpoint(
        path_to_save_model,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )

    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

    model = signs_model(nbr_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator,
              epochs=epochs, 
              batch_size=batch_size, 
              validation_data=val_generator,
              callbacks=[chkpt_saver, early_stop]
            )