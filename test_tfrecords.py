from matplotlib import image
import numpy as np
import tensorflow as tf
from utils.datagen import parse_record, decode_record, get_dataset

# dataset = tf.data.TFRecordDataset('./data_project/train/SN_6.tfrecords')

# im_list = []
# n_samples_to_show = 16
# c = 0
# for record in dataset:
#     c+=1
#     if c > n_samples_to_show:
#         break
#     parsed_record = parse_record(record)
#     decoded_record = decode_record(parsed_record)
#     image, label = decoded_record
#     im_list.append(image)

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
# fig = plt.figure(figsize=(4,4))
# grid = ImageGrid(fig, 111, nrows_ncols=(4,4), axes_pad=0.1)

# for ax, im in zip(grid, im_list):
#     ax.imshow(im)

training_data = get_dataset('./data_project/train/SN_6.tfrecords')
image_batch, label_batch = next(iter(training_data))

print(image_batch.shape)