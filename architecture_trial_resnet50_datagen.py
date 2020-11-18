
'''
imports and global
'''
import utils.helper as helper
import numpy as np

import tensorflow as tf
from tensorflow import keras
# from keras_tqdm import TQDMCallback
# from keras.callbacks import ModelCheckpoint

import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
SM_FRAMEWORK = os.getenv('SM_FRAMEWORK')
import segmentation_models as sm
sm.set_framework(SM_FRAMEWORK)

import wandb
from wandb.keras import WandbCallback

from utils.datagen import get_dataset

''' 
---------------------------------------
GLOBAL - CHANGE HERE
--------------------------------------- 
''' 

BACKBONE = 'resnet50'
wandb.init(project='architecture_trial_resnet50_datagen')
model_name = 'architecture_trial_resnet50_datagen'

print('available gpus')
print(tf.config.experimental.list_physical_devices('GPU'))
gpu = tf.config.experimental.list_physical_devices('GPU')[0]

print('allowing GPU memory growth')
tf.config.experimental.set_memory_growth(gpu, True)

'''
loading data in the form of tf.data.dataset
'''
PATH_RESULTS, PATH_HISTORIES, PATH_FIGURES, PATH_CHECKPOINTS, PATH_PREDICTIONS = helper.results_paths()

print('reading tf.data.Dataset')
train_data = get_dataset('./data_project/train/SN_6.tfrecords', train=True)
val_data = get_dataset('./data_project/train/SN_6_val.tfrecords', train=False)
test_data = get_dataset('./data_project/test/SN_6_test.tfrecords', train=False)
print("tf.data.Dataset for train/val/test read")


'''
define the model - make sure to set model name
'''
model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(None, None, 3))
model.compile(
    optimizer='adam',
    loss=sm.losses.BinaryFocalLoss(alpha=0.75, gamma=0.25),
    metrics=[sm.metrics.IOUScore()],
)


'''
fit model - save best weights at each epoch
'''
# CheckpointCallback = ModelCheckpoint(str(PATH_CHECKPOINTS / (model_name + '.hdf5')), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='auto', period=1)

history = model.fit(
   train_data,
#    epochs=1,
   epochs=100,
   validation_data=val_data,
   callbacks=[
    #    TQDMCallback(),
       WandbCallback(log_weights=True, save_weights_only=True),
    #    CheckpointCallback
       ]
)

helper.history_saver(history, model_name, PATH_HISTORIES, already_npy=False)
history = helper.history_loader(model_name, PATH_HISTORIES)
helper.plot_metrics(history, model_name, PATH_FIGURES)


'''
predict on the test set. load best weights from checkpoints
'''
model.load_weights(str(PATH_CHECKPOINTS / (model_name + '.hdf5')))

# predictions = model.predict(
#     X_test,
#     verbose=1,
#     callbacks=[
#         TQDMCallback()
#     ]
# ) 

test_metrics = model.evaluate(test_data, steps=3)

test_metrics_dict = {
    'test_loss': test_metrics[0],
    'test_iou_score': test_metrics[1]
}

# np.save(PATH_PREDICTIONS / model_name, predictions)
np.save(PATH_PREDICTIONS/str(model_name + "_prediction_score"), test_metrics_dict)