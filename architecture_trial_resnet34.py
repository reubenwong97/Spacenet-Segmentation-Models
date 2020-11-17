
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

BACKBONE = 'resnet34'
wandb.init(project='architecture_trial_resnet34')
model_name = 'architecture_trial_resnet34'





'''
load your data. this is a 5GB numpy array with all our data
'''
print("loading data")
PATH_RESULTS, PATH_HISTORIES, PATH_FIGURES, PATH_CHECKPOINTS, PATH_PREDICTIONS = helper.results_paths()
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
'''
CheckpointCallback = ModelCheckpoint(str(PATH_CHECKPOINTS / (model_name + '.hdf5')), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='auto', period=1)

history = model.fit(
   x=X_train,
   y=Y_train,
   batch_size=128,
   epochs=100,
   validation_split=0.3,
   callbacks=[
       TQDMCallback(),
       WandbCallback(log_weights=True),
       CheckpointCallback
       ]
)



'''
predict on the test set
'''
predictions = model.predict(
    X_test,
    verbose=1,
    callbacks=[
        TQDMCallback()
    ]
)

test_metrics = model.evaluate(X_test, Y_test, batch_size=64)

test_metrics_dict = {
    'test_loss': test_metrics[0],
    'test_iou_score': test_metrics[1]
}


'''
save the results and load 
'''
helper.history_saver(history, model_name, PATH_HISTORIES, already_npy=False)
history = helper.history_loader(model_name, PATH_HISTORIES)
helper.plot_metrics(history, model_name, PATH_FIGURES)

np.save(PATH_PREDICTIONS / model_name, predictions)
np.save(PATH_PREDICTIONS/str(model_name + "_prediction_score"), test_metrics_dict)