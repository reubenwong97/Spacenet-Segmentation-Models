import wandb
from tensorflow import keras
import utils.helper as helper

# this is a 5GB numpy array with all our data
X_train, Y_train, X_test, Y_test = helper.generate_train_test()

# potentially perform CV, to see

# inspect picture to test that load works
index = 8
helper.plot_img_mask(index, X_train[index], Y_train[index])

wandb.init(project='spacenet_6_example')
config = wandb.config

# set decided params here -> helps with neatness and we only change it at one place
config.num_neurons = 10
config.lr = 1e-5
config.input_shape = (500,1)
config.batch_size = 32

x = [1, 2, 3]
y = [0.5, 1, 1.5]

# helps log anything u want to log
# wandb.log({"loss": loss})

# anything we may want to pass as a parameter, use the config object
model = keras.Sequential([keras.layers.Dense(config.num_neurons, config.input_shape)])
model.compile(optimizer=keras.optimizers.Adam(config.lr), loss=keras.losses.SparseCategoricalCrossEntropy())
# more settings to explore in WandbCallback
model.fit(x, y, batch_size=config.batch_size, callbacks=[wandb.WandbCallback(log_weights=True)])