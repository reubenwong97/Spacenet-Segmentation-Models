
'''
imports and global
'''
import utils.helper as helper

import tensorflow as tf
from tensorflow import keras
from keras_tqdm import TQDMCallback
from keras.callbacks import ModelCheckpoint

import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
SM_FRAMEWORK = os.getenv('SM_FRAMEWORK')
import segmentation_models as sm
sm.set_framework(SM_FRAMEWORK)

import wandb
from wandb.keras import WandbCallback


''' 
---------------------------------------
GLOBAL - CHANGE HERE
--------------------------------------- 
''' 

BACKBONE = 'vgg19'
wandb.init(project='architecture_trial_vgg19')
model_name = 'architecture_trial_vgg19'





'''
load your data. this is a 5GB numpy array with all our data
'''
print("loading data")
PATH_RESULTS, PATH_HISTORIES, PATH_FIGURES, PATH_CHECKPOINTS = helper.results_paths()
X_train, Y_train, X_test, Y_test = helper.generate_train_test()
print("X_train, Y_train, X_test, Y_test loaded")


'''
preprocess input to ensure it fits the model definition
'''
print("preprocessing input")
preprocess_input = sm.get_preprocessing(BACKBONE)

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

X_train = tf.dtypes.cast(X_train, tf.dtypes.float32)
X_test = tf.dtypes.cast(X_test, tf.dtypes.float32)
Y_train = tf.dtypes.cast(Y_train, tf.dtypes.float32)
Y_test = tf.dtypes.cast(Y_test, tf.dtypes.float32)
print("finished preprocessing input")


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
if you use data generator use model.fit_generator(...) instead of model.fit(...)
more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
'''
CheckpointCallback = ModelCheckpoint(str(PATH_CHECKPOINTS / (model_name + '.hdf5')), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='auto', period=1)

history = model.fit(
   x=X_train,
   y=Y_train,
   batch_size=64,
   epochs=100,
   validation_split=0.3,
   callbacks=[
       TQDMCallback(),
       WandbCallback(log_weights=True),
       CheckpointCallback
       ]
)


'''
save the results and load 
'''
helper.history_saver(history, model_name, PATH_HISTORIES, already_npy=False)
history = helper.history_loader(model_name, PATH_HISTORIES)
