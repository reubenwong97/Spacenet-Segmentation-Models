
from tensorflow import keras
import utils.helper as helper
from keras_tqdm import TQDMCallback

import segmentation_models as sm


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# load your data
# this is a 5GB numpy array with all our data
X_train, Y_train, X_test, Y_test = helper.generate_train_test()

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
model.fit(
   x=X_train,
   y=Y_train,
   batch_size=32,
   epochs=100,
   validation_split=0.2,
   callbacks=[TQDMCallback()]
)