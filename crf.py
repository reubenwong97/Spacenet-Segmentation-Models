
'''
imports and global
'''
import utils.helper as helper
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras_tqdm import TQDMCallback
from keras.callbacks import ModelCheckpoint

import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
SM_FRAMEWORK = os.getenv('SM_FRAMEWORK')
import segmentation_models_dev as sm
sm.set_framework(SM_FRAMEWORK)

import wandb
from wandb.keras import WandbCallback

from utils.datagen import get_dataset

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

        
''' 
---------------------------------------
GLOBAL - CHANGE HERE
input the best model to use based on test data, because we are visualizing the test data
run results/predictions_scores.py to find out what the best model is for test data

--> data_augmentation_false is the best 
--------------------------------------- 
''' 
model_name = 'data_augmentation_false' # from data_augmentation
augment = False # from data_augmentation

decoder_drop_rate = 0.0 # from internal_parameter_decoderdroprate
decoder_use_batchnorm=False # from internal_parameter_decodernorm
decoder_use_groupnorm = True # from internal_parameter_decodernorm
decoder_groupnorm_groups = 8 # from internal_parameter_decodernorm
backbone = 'resnet18'  # from internal_parameter_activation
encoder_activation = 'relu' # from internal_parameter_activation


'''
loading data in the form of tf.data.dataset
'''
PATH_RESULTS, PATH_HISTORIES, PATH_FIGURES, PATH_CHECKPOINTS, PATH_PREDICTIONS, PATH_SAMPLE_FIGS = helper.results_paths()

print('reading tf.data.Dataset')
train_data = get_dataset('./data_project/train/SN_6.tfrecords', augment=augment)
val_data = get_dataset('./data_project/train/SN_6_val.tfrecords')
test_data = get_dataset('./data_project/test/SN_6_test.tfrecords')
print("tf.data.Dataset for train/val/test read")


'''
define the model - make sure to set model name
'''
model = sm.Unet(backbone, encoder_weights='imagenet', input_shape=(None, None, 3),
    decoder_block_type='upsampling', decoder_drop_rate=decoder_drop_rate,
    decoder_use_batchnorm=decoder_use_batchnorm, decoder_use_groupnorm=decoder_use_groupnorm, decoder_groupnorm_groups=decoder_groupnorm_groups,
    encoder_activation=encoder_activation
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
    loss=sm.losses.JaccardLoss(),
    metrics=[sm.metrics.IOUScore()],
)


'''
predict on a subset of the test set. load best weights from checkpoints
'''
model.load_weights(str(PATH_CHECKPOINTS / (model_name + '.hdf5')))

image_batch, mask_batch = next(iter(test_data))

predictions = model.predict(
    image_batch,
    verbose=1,
    callbacks=[
        TQDMCallback()
    ]
)

test_metric = sm.metrics.IOUScore()


'''
plot and save the plots 
'''
for index in range(len(predictions)):
    save_path = PATH_SAMPLE_FIGS/ ('sample_'+str(index)) 
    helper.plot_img_mask(index, image_batch[index], mask_batch[index], pred=predictions[index], save_path=save_path, display=False)

