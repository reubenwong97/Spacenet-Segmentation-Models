import numpy as np
import tensorflow.keras as keras
from .helper import data_paths

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=64, dim=(224, 224), n_channels=3, shuffle=True, n_classes=2,
                    train=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train = train
        self.on_epoch_end()
        self.PATH_TRAIN_IMG, self.PATH_TRAIN_MASK, self.PATH_TEST_IMG, self.PATH_TEST_MASK = data_paths()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=bool)

        for i, ID in enumerate(list_IDs_temp):
            if self.train:
                X[i,] = np.load(self.PATH_TRAIN_IMG + ID)
                y[i,] = np.load(self.PATH_TRAIN_MASK + ID)
            else:
                X[i,] = np.load(self.PATH_TEST_IMG + ID)
                y[i,] = np.load(self.PATH_TEST_MASK + ID)

        return X, y