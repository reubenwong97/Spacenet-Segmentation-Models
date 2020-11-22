class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_fname_list, mask_fname_list, img_path, mask_path, rebuild_func, batch_size=128, shuffle=True, test_gen=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_fname_list = img_fname_list
        self.mask_fname_list = mask_fname_list
        self.IMG_PATH = img_path
        self.MASK_PATH = mask_path
        self.rebuild_func = rebuild_func
        self.test_gen = test_gen

        self.data_len = len(self.img_fname_list)

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.data_len) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        img_list = [self.img_fname_list[k] for k in indexes]
        mask_list = [self.mask_fname_list[k] for k in indexes]

        X, Y = self.__data_generation(img_list, mask_list)

        if not self.test_gen:
            return X, Y
        else:
            return X
        

    def on_epoch_end(self):
        self.indexes = np.arange(self.data_len)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_list, mask_list):
        X, Y = [],[]

        for fname in img_list:
            npy = self.rebuild_func(self.IMG_PATH/fname)
            X.append(npy)
        
        for fname in mask_list:
            npy = self.rebuild_func(self.MASK_PATH/fname)
            Y.append(npy)
            
        return tf.dtypes.cast(X, tf.dtypes.float32), tf.dtypes.cast(Y, tf.dtypes.float32)