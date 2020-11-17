import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

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

def DatasetFromSequenceClass(sequenceClass, stepsPerEpoch, nEpochs=1, batchSize=128, dims=[224,224,3], n_classes=2, data_type=tf.float32, label_type=tf.float32):
    # eager execution wrapper
    def DatasetFromSequenceClassEagerContext(func):
        def DatasetFromSequenceClassEagerContextWrapper(batchIndexTensor):
            # Use a tf.py_function to prevent auto-graph from compiling the method
            tensors = tf.py_function(
                func,
                inp=[batchIndexTensor],
                Tout=[data_type, label_type]
            )

            # set the shape of the tensors - assuming channels last
            tensors[0].set_shape([batchSize, dims[0], dims[1], dims[2]])   # [samples, height, width, nChannels]
            tensors[1].set_shape([batchSize, dims[0], dims[1]]) # [samples, height, width, nClasses for one hot]
            # tensors[1].set_shape([batchSize, dims[0], dims[1], n_classes]) # [samples, height, width, nClasses for one hot]
            return tensors
        return DatasetFromSequenceClassEagerContextWrapper

    # TF dataset wrapper that indexes our sequence class
    @DatasetFromSequenceClassEagerContext
    def LoadBatchFromSequenceClass(batchIndexTensor):
        # get our index as numpy value - we can use .numpy() because we have wrapped our function
        batchIndex = batchIndexTensor.numpy()

        # zero-based index for what batch of data to load; i.e. goes to 0 at stepsPerEpoch and starts cound over
        zeroBatch = batchIndex % stepsPerEpoch

        # load data
        data, labels = sequenceClass[zeroBatch]

        # convert to tensors and return
        return tf.convert_to_tensor(data), tf.convert_to_tensor(labels)

    # create our data set for how many total steps of training we have
    dataset = tf.data.Dataset.range(stepsPerEpoch*nEpochs)

    # return dataset using map to load our batches of data, use TF to specify number of parallel calls
    return dataset.map(LoadBatchFromSequenceClass, num_parallel_calls=tf.data.experimental.AUTOTUNE)