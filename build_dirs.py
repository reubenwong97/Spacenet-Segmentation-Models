import os

if not os.isdir('./data_project'):
    os.makedirs('./data_project')
    if not os.isdir('./data_project/train'):
        os.makedirs('./data_project/train')
    if not os.isdir('./data_project/test'):
        os.makedirs('./data_project/test')