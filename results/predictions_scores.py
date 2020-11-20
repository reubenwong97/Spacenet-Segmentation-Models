import numpy as np
import os

fnames = next(os.walk('./predictions'))[2]
fnames = [x for x in fnames if x[-4:] == '.npy']

best_model = ['', 0, 0]

for fname in fnames:
    if "decoderdroprate" in fname:
        score = np.load('./predictions/'+fname, allow_pickle=True)
        score = score.item()
        print(f'{fname}: \n{score}\n')

        if score['test_iou_score'] > best_model[2]:
            best_model[0] = fname
            best_model[1] = score['test_loss']
            best_model[2] = score['test_iou_score']

print(f"best model: {best_model}")
