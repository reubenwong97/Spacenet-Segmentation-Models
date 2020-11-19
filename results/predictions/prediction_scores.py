import numpy as np
import os

fnames = next(os.walk('.'))[2]
fnames = [x for x in fnames if x[-4:] == '.npy']

for fname in fnames:
    if "optimizer" in fname:
        score = np.load(fname, allow_pickle=True)
        print(f'{fname}: \n{score}\n')