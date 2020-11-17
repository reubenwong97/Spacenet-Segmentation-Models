import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import utils.helper as helper

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=64, dim=(224, 224), n_channels=3, shuffle=True, n_classes=2,
                    train=True, validation=True):
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train = train
        self.validation = validation

        self.PATH_TRAIN_IMG, self.PATH_TRAIN_MASK, self.PATH_TEST_IMG, self.PATH_TEST_MASK = helper.data_paths()        
        self.data_len = len(helper.get_fnames(self.PATH_TRAIN_IMG)) if self.train else len(helper.get_fnames(self.PATH_TEST_IMG))

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        if self.train:
            self.img_fname = helper.get_fnames(self.PATH_TRAIN_IMG)
            self.mask_fname = helper.get_fnames(self.PATH_TRAIN_MASK)
        else:
            self.img_fname = helper.get_fnames(self.PATH_TEST_IMG)
            self.mask_fname = helper.get_fnames(self.PATH_TEST_MASK)

        img_list = [self.img_fname[k] for k in indexes]
        mask_list = [self.mask_fname[k] for k in indexes]

        X, y = self.__data_generation(img_list, mask_list)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.data_len)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_list, mask_list):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=bool)

        for i, fname in enumerate(img_list):
            if self.train:
                X[i,] = np.load(self.PATH_TRAIN_IMG + fname)
            else:
                X[i,] = np.load(self.PATH_TEST_IMG + fname)

        for i, fname in enumerate(mask_list):
            if self.train:
                y[i,] = np.load(self.PATH_TRAIN_MASK + fname)
            else:
                y[i,] = np.load(self.PATH_TEST_MASK + fname)

        return tf.dtypes.cast(X, tf.dtypes.float32), tf.dtypes.cast(y, tf.dtypes.float32)