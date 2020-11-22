# README
## Logging
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
If we are using keras, we can make use of wandb callback.
```python
from wandb.keras import WandbCallback
history = model.fit(
  train, 
  epochs=1,
  callbacks=[WandbCallback()]
)
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
