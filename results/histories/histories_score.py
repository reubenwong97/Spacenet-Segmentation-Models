import numpy as np
import os

fnames = next(os.walk('.'))[2]
fnames = [x for x in fnames if x[-4:] == '.npy']

for fname in fnames:
    if "external" in fname:
        history = np.load(fname, allow_pickle=True)
        history = history.item()
        
        index_min = np.argmin(history['val_loss'])
        min_val_loss = history['val_loss'][index_min]
        max_val_iou_score = history['val_iou_score'][index_min]

        print(f'{fname}: \nMin val_loss: {min_val_loss}, Max val_iou_score:{max_val_iou_score}\n')