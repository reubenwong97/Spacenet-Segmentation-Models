
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
import segmentation_models as sm
sm.set_framework(SM_FRAMEWORK)

import wandb
from wandb.keras import WandbCallback


''' 
---------------------------------------
GLOBAL - CHANGE HERE
--------------------------------------- 
''' 

BACKBONE = 'seresnet50'
wandb.init(project='architecture_trial_seresnet50')
model_name = 'architecture_trial_seresnet50'





'''
Creating train, val, test generators
'''
print("creating generators")
PATH_RESULTS, PATH_HISTORIES, PATH_FIGURES, PATH_CHECKPOINTS, PATH_PREDICTIONS = helper.results_paths()

train_generator, val_generator, test_generator = helper.generate_train_val_test()
print("Generators created")


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
   train_generator,
   validation_data=val_generator,
   epochs=100,
   callbacks=[
       TQDMCallback(),
       WandbCallback(log_weights=True),
       CheckpointCallback
       ]
)

predictions = model.predict(
    test_generator,
    verbose=1,
    callbacks=[
        TQDMCallback()
    ]
)


'''
save the results and load 
'''
helper.history_saver(history, model_name, PATH_HISTORIES, already_npy=False)
history = helper.history_loader(model_name, PATH_HISTORIES)
helper.plot_metrics(history, model_name, PATH_FIGURES)

np.save(PATH_PREDICTIONS / model_name, predictions)
