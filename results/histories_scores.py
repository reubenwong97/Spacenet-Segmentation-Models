import numpy as np
import os

fnames = next(os.walk('./histories'))[2]
fnames = [x for x in fnames if x[-4:] == '.npy']

best_model = ['', 0, 0]

for fname in fnames:
    if "decodernorm" in fname:
        history = np.load('./histories/'+fname, allow_pickle=True)
        history = history.item()
        
        index_min = np.argmin(history['val_loss'])
        min_val_loss = history['val_loss'][index_min]
        max_val_iou_score = history['val_iou_score'][index_min]

        print(f'{fname}: \nMin val_loss: {min_val_loss}, Max val_iou_score:{max_val_iou_score}\n')

        if max_val_iou_score > best_model[2]:
            best_model[0] = fname
            best_model[1] = min_val_loss
            best_model[2] = max_val_iou_score

print(f"best model: {best_model}")
