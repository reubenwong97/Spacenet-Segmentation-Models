import wandb
from tensorflow import keras

wandb.init(project='spacenet_6_example')
config = wandb.config

# set decided params here
config.num_neurons = 10
config.lr = 1e-5
config.input_shape = (500,1)
config.batch_size = 32

x = [1, 2, 3]
y = [0.5, 1, 1.5]

model = keras.Sequential([keras.layers.Dense(config.num_neurons, config.input_shape)])
model.compile(optimizer=keras.optimizers.Adam(config.lr), loss=keras.losses.SparseCategoricalCrossEntropy())
# more settings to explore in WandbCallback
model.fit(x, y, batch_size=config.batch_size, callbacks=[wandb.WandbCallback(log_weights=True)])