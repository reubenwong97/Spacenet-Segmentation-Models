
import tensorflow as tf
from tensorflow import keras
import utils.helper as helper
from keras_tqdm import TQDMCallback

import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
SM_FRAMEWORK = os.getenv('SM_FRAMEWORK')

import segmentation_models as sm
sm.set_framework(SM_FRAMEWORK)
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)

import wandb
wandb.init(project='spacenet_6_trial_run')


# load your data
# this is a 5GB numpy array with all our data
print("loading data")
X_train, Y_train, X_test, Y_test = helper.generate_train_test()
print("X_train, Y_train, X_test, Y_test loaded")

# preprocess input
print("preprocessing input")
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

X_train = tf.dtypes.cast(X_train, tf.dtypes.float32)
X_test = tf.dtypes.cast(X_test, tf.dtypes.float32)
Y_train = tf.dtypes.cast(Y_train, tf.dtypes.float32)
Y_test = tf.dtypes.cast(Y_test, tf.dtypes.float32)
print("finished preprocessing input")


print(X_train.dtype)
print(X_test.dtype)
print(Y_train.dtype)
print(Y_test.dtype)


# input subset of data only
# X_train, Y_train, X_test, Y_test = X_train[:100], Y_train[:100], X_test[:100], Y_test[:100]

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(None, None, 3))
model.compile(
    optimizer='adam',
    loss=sm.losses.BinaryFocalLoss(alpha=0.75, gamma=0.25),
    metrics=[sm.metrics.IOUScore()],
)

# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
model.fit(
   x=X_train,
   y=Y_train,
   batch_size=32,
   epochs=1000,
   validation_split=0.3,
   callbacks=[TQDMCallback()]
)