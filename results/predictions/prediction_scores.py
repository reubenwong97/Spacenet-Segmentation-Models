import numpy as np

model_names = ['efficientnetb0', 'resnet34', 'resnet50']

for model_name in model_names:
    score = np.load('architecture_trial_' + model_name + '_prediction_score.npy', allow_pickle=True)
    print(f'{model_name}: {score}')
