'''
imports
'''
from pathlib import Path
from matplotlib import pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import numpy as np
from .datagen import DataGenerator

'''
# used to generate the PosixPath variables for various common paths
# only works if the calling file exists in the root directory, that is, can access the folder data_project
'''
def data_paths():
    ROOT_DIR = Path('')

    PATH_DATA_PROJECT = ROOT_DIR/'data_project'

    PATH_TRAIN = PATH_DATA_PROJECT/'train'
    PATH_TEST = PATH_DATA_PROJECT/'test'

    PATH_TRAIN_IMG = PATH_TRAIN/'img'
    PATH_TRAIN_MASK = PATH_TRAIN/'mask'
    PATH_TEST_IMG = PATH_TEST/'img'
    PATH_TEST_MASK = PATH_TEST /'mask'

    return PATH_TRAIN_IMG, PATH_TRAIN_MASK, PATH_TEST_IMG, PATH_TEST_MASK


'''
used to generate the PosixPath variables for the results to save
'''
def results_paths():
    ROOT_DIR = Path('')
    PATH_RESULTS = ROOT_DIR /'results'
    PATH_HISTORIES = PATH_RESULTS / 'histories'
    PATH_FIGURES = PATH_RESULTS / 'figures'
    PATH_CHECKPOINTS = PATH_RESULTS / 'checkpoints'
    PATH_PREDICTIONS = PATH_RESULTS / 'predictions'

    return PATH_RESULTS, PATH_HISTORIES, PATH_FIGURES, PATH_CHECKPOINTS, PATH_PREDICTIONS


''' 
used to save the history of a model as a npy file
'''
# filename like 'history/model_name.npy'
def history_saver(history, model_name, history_save_path, already_npy=False):
  history_json = {}

  if already_npy:
    history_npy = history

  else:
    history_npy = history.history

  np.save(history_save_path/model_name, history_npy)
  print("History saved")



''' 
used to load the history of a model from a npy file
'''
# filename like 'history/model_name.npy'
def history_loader(model_name, history_save_path):
  history_save_path = history_save_path/str(model_name+'.npy')
  history=np.load(history_save_path,allow_pickle='TRUE').item()
  print('History loaded')
  
  return history 



'''
used to plot the image, and the mask side by side, and also the prediction, if any
index: int, img: np.ndarray, mask: np.ndarray, pred: np.ndarray
'''
def plot_img_mask(index, img, mask, pred=None):
    if pred == None:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
        ax1.imshow(img)
        ax1.set_title('image')
        ax2.imshow(mask)
        ax2.set_title('ground truth mask')
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(21,7))
        ax1.imshow(img)
        ax1.set_title('image')
        ax2.imshow(mask)
        ax2.set_title('ground truth mask')
        ax3.imshow(pred)
        ax3.set_title('predicted mask')
    
    print("Index: {}".format(index))
    plt.show()


'''
used to plot the metrics for a given history
'''
def plot_metrics(history, model_name, figure_save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # plot losses
    train_loss = history['loss']
    val_loss = history['val_loss']
    loss_title = 'loss against epochs'

    ax1.plot(train_loss, label='train')
    ax1.plot(val_loss, label='val')
    ax1.set_title(loss_title)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax1.legend()

    # plot iou_score
    iou_score = history['iou_score']
    val_iou_score = history['val_iou_score']
    iou_score_title = 'iou_score against epochs'

    ax2.plot(iou_score, label='train')
    ax2.plot(val_iou_score, label='val')
    ax2.set_title(iou_score_title)
    ax2.set_ylabel('iou_score')
    ax2.set_xlabel('epochs')
    ax2.legend()

    # save figure
    fig.suptitle('Metrics for model: ' + model_name)
    plt.savefig(figure_save_path/f'{model_name}.png')

    plt.show()


'''    
used to obtain all the filenames in a given directory as a list
path: PosixPath
'''
def get_fnames(path):
    fnames = next(os.walk(path))[2]
    fnames.sort()
    return fnames


'''
used to reconstruct a numpy array representation of an image from raveled .npy files
npy_path: PosixPath, img_height: int=224, img_width: int=224
'''
def rebuild_npy(npy_path, img_height=224, img_width=224):
    img_npy = np.load(npy_path)
    img_channel = int(len(img_npy)/img_height/img_width)
    
    if img_channel == 1:
        return img_npy.reshape(img_height, img_width)
    elif img_channel == 3:
        return img_npy.reshape(img_height, img_width, img_channel)
    else:
        print("cannot rebuild numpy array")
        return
            
    # return img_npy.reshape(img_height, img_width, img_channel)


'''
used to generate X_train, Y_train, X_test, Y_test as numpy arrays, from their .npy files
'''
def generate_train_test():
    paths = data_paths()
    data = [[], [], [], []]

    for index, path in tqdm(enumerate(paths), total=len(paths)):
        fnames = get_fnames(path)
        
        # for fname in tqdm(fnames[:128], total=len(fnames[:128])):
        for fname in tqdm(fnames, total=len(fnames)):
            npy = rebuild_npy(path / fname)
            data[index].append(npy)

        data[index] = np.array(data[index])
    
    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    
    return (X_train, Y_train, X_test, Y_test)


'''
used to generate X_train, Y_train, X_test, Y_test as numpy arrays, from their .npy files
data generator style
'''

# def generate_train_val_test(val_percent=0.7):
#     PATH_TRAIN_IMG, PATH_TRAIN_MASK, PATH_TEST_IMG, PATH_TEST_MASK = data_paths()

#     val_split = int(len(get_fnames(PATH_TRAIN_IMG))*val_percent)
#     X_train_fnames = get_fnames(PATH_TRAIN_IMG)[:val_split]
#     Y_train_fnames = get_fnames(PATH_TRAIN_MASK)[:val_split]
#     X_val_fnames = get_fnames(PATH_TRAIN_IMG)[val_split:]
#     Y_val_fnames = get_fnames(PATH_TRAIN_MASK)[val_split:]
#     X_test_fnames = get_fnames(PATH_TEST_IMG)
#     Y_test_fnames = get_fnames(PATH_TEST_MASK)

#     train_generator = DataGenerator(X_train_fnames, Y_train_fnames, PATH_TRAIN_IMG, PATH_TRAIN_MASK, rebuild_func=rebuild_npy)
#     val_generator = DataGenerator(X_val_fnames, Y_val_fnames, PATH_TRAIN_IMG, PATH_TRAIN_MASK, rebuild_func=rebuild_npy)
#     test_generator = DataGenerator(X_test_fnames, Y_test_fnames, PATH_TEST_IMG, PATH_TEST_MASK, rebuild_func=rebuild_npy)

    
#     # train_generator = DataGenerator(X_train_fnames[:64], Y_train_fnames[:64], PATH_TRAIN_IMG, PATH_TRAIN_MASK, rebuild_func=rebuild_npy)
#     # val_generator = DataGenerator(X_val_fnames[:64], Y_val_fnames[:64], PATH_TRAIN_IMG, PATH_TRAIN_MASK, rebuild_func=rebuild_npy)
#     # test_generator = DataGenerator(X_test_fnames[:64], Y_test_fnames[:64], PATH_TEST_IMG, PATH_TEST_MASK, rebuild_func=rebuild_npy, test_gen=True)


#     return train_generator, val_generator, test_generator
