
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

''' 
---------------------------------------
GLOBAL - CHANGE HERE
--------------------------------------- 
''' 

BACKBONE = 'efficientnetb5'
wandb.init(project='architecture_trial_efficientnetb5_datagen')
model_name = 'architecture_trial_efficientnetb5_datagen'

print('available gpus')
print(tf.config.experimental.list_physical_devices('GPU'))
gpu = tf.config.experimental.list_physical_devices('GPU')[0]

print('allowing GPU memory growth')
tf.config.experimental.set_memory_growth(gpu, True)

'''
load your data. this is a 5GB numpy array with all our data
'''
print("loading data")
# PATH_RESULTS, PATH_HISTORIES, PATH_FIGURES, PATH_CHECKPOINTS, PATH_PREDICTIONS = helper.results_paths()
# X_train, Y_train, X_test, Y_test = helper.generate_train_test()
print("X_train, Y_train, X_test, Y_test loaded")


'''
preprocess input to ensure it fits the model definition
'''
print("preprocessing input")
# preprocess_input = sm.get_preprocessing(BACKBONE)

# X_train = preprocess_input(X_train)
# X_test = preprocess_input(X_test)

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

optimizer = tf.keras.optimizer.Adam()
loss_object = sm.losses.BinaryFocalLoss(alpha=0.75, gamma=0.25)
metrics = sm.metrics.IOUScore()

train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

train_iou = tf.keras.metrics.Mean(name='train_iou')
val_iou = tf.keras.metrics.Mean(name='val_iou')
test_iou = tf.keras.metrics.Mean(name='test_iou')

'''
fit model - save best weights at each epoch
'''
# CheckpointCallback = ModelCheckpoint(str(PATH_CHECKPOINTS / (model_name + '.hdf5')), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='auto', period=1)
# CheckpointCallback = ModelCheckpoint(filepath=str(PATH_CHECKPOINTS/model_name), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='auto')

# history = model.fit(
#    train_data,
#    epochs=100,
#    validation_data=val_data,
#    callbacks=[
#        TQDMCallback(),
#        WandbCallback(log_weights=True),
#        CheckpointCallback
#        ]
# )

# helper.history_saver(history, model_name, PATH_HISTORIES, already_npy=False)
# history = helper.history_loader(model_name, PATH_HISTORIES)
# helper.plot_metrics(history, model_name, PATH_FIGURES)

def train_step(model, img, mask):
    with tf.GradientTape() as tape:
        out = model(img)
        loss = loss_object(out, mask)
        metric = metrics(out, mask)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    train_loss(loss)
    train_iou(metric)

def val_step(model, img, mask):
    out = model(img)
    t_loss = loss_object(mask, out)

    val_loss(t_loss)
    val_iou(mask, out)

train_iou, val_iou , train_loss_metric, val_loss_metric = [], [], [], []
for epoch in range(100):
    train_loss.reset_states()
    train_iou.reset_states()
    val_loss.reset_states()
    val_iou.reset_states()
    
    for image_batch, mask_batch in train_data:
        train_step(model, image_batch, mask_batch)

    for image_batch, mask_batch in val_data:
        val_step(model, image_batch, mask_batch)

    train_iou.append(train_iou.result())
    val_iou.append(val_iou.result())
    train_loss_metric.append(train_loss.result())
    val_loss_metric.append(val_loss.result())

    template = 'Epoch {}, Loss: {}, IOU: {}, Val Loss: {}, Val IOU: {}'
    print (template.format(epoch+1,
                          train_loss.result(),
                          train_iou.result(),
                          val_loss.result(),
                          val_iou.result()))

    if epoch % 20 == 0:
        model_save_template = 'efficientnetb5_{}_{}'
        

if not os.isdir('./results/raw'):
    os.makedirs('./results/raw')

np.save('./results/raw/efficientnetb5_train_iou.npy', train_iou)
np.save('./results/raw/efficientnetb5_val_iou.npy', val_iou)
np.save('./results/raw/efficientnetb5_train_loss.npy', train_loss_metric)
np.save('./results/raw/efficientnetb5_val_loss.npy', val_loss_metric)

'''
predict on the test set. load best weights from checkpoints
'''
# model.load_weights(str(PATH_CHECKPOINTS / (model_name)))

# predictions = model.predict(
#     X_test,
#     verbose=1,
#     callbacks=[
#         TQDMCallback()
#     ]
# ) 

# test_metrics = model.evaluate(test_data, steps=3)

# test_metrics_dict = {
#     'test_loss': test_metrics[0],
#     'test_iou_score': test_metrics[1]
}

# np.save(PATH_PREDICTIONS / model_name, predictions)
# np.save(PATH_PREDICTIONS/str(model_name + "_prediction_score"), test_metrics_dict)