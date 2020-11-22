# README
## Logging
### Params Handler
For easy tracking, parameters are all handled with the `Params` class defined in `logging.py`. It allows us to load parameters neatly from a json file. For example, we can do this:
```json
{
    "params_file": "default.json",
    "log_file": "train.log",

    "lr": 0.0003,
    "seed": 1,
    "batch_size": 64,
    "num_steps": 365,
    "hidden_size": [32, 64,256],
    "cuda": true
}
```
We can then access the params throughout the code in a dict-like manner like this:
```python
params = Params(path_to_params)
# get learning rate
lr = params.lr
```
### Weights and Biases (Doper)
I will create an api key and we will then log to the same project. Weights and biases provides better graphics, inspection of gradients, as well as hyperparameter tuning, with the ability to perform automatic sweeps through the hyperparameter space. 
First, install wandb and attach api key:
```bash
pip install wandb
wandb login
```
Next, intialise it in the training scripts:
```python
import wandb
wandb.init(project="spacenet6")
```
Then, we need to declare the hyperparameters we want to use:
```python
wandb.config.dropout = 0.2
wandb.config.hidden_layer_size = 128
```
We log metrics similarly to tensorboard:
```python
def my_train_loop():
    for epoch in range(10):
        loss = 0 # change as appropriate :)
        wandb.log({'epoch': epoch, 'loss': loss})
```
When we want to perform sweeps, we need to make sure the metric that we are using to measure our experiment has been logged like the above. Then, we configure the sweep with a yaml file, I'll probably do this because my account is needed. 
```yaml
program: train.py
command:
  - ${env}
  - ${interpreter}
  - ${program}
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.001
    max: 0.1
  optimizer:
    values: ["adam", "sgd"]
```
**Note:** This section is important in the configuration if we don't use argparsers:
```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
There's a whole bunch of available configurations [here](https://docs.wandb.com/sweeps/configuration)!
### Tensorboard (Outdated) - Funky Logging
As much as possible should be logged when conducting experiments. We should probably only need to do this within a `main.py` file but we shall see. See some quick steps for logging below:
```python
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
```
HParams allows us to do nifty stuff like set a range of values we want to try so it is easy for us to iterate over them in our experiments. Example configuration below. 
```python
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
```
We can access the parameters similar to the `Params` class in our `logging.py` file.
```python
def train_test_model(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  ])
  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

  model.fit(x_train, y_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
  _, accuracy = model.evaluate(x_test, y_test)
  return accuracy
```
For more details on tensorboard please see [here](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams).
