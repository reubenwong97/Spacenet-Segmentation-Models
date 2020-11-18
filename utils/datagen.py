from utils.helper import rebuild_npy, data_paths, get_fnames
from utils.data_aug import data_augment
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from functools import partial

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def construct_records(tfrecords_filename='./data_project/train/SN_6.tfrecords', test=False):
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    PATH_TRAIN_IMG, PATH_TRAIN_MASK, PATH_TEST_IMG, PATH_TEST_MASK = data_paths()

    if test:
        img_names = get_fnames(PATH_TEST_IMG)
        mask_names = get_fnames(PATH_TEST_MASK)

        for i in range(len(img_names)):
            image_full_path = PATH_TEST_IMG/img_names[i]
            mask_full_path = PATH_TEST_MASK/mask_names[i]

            img = rebuild_npy(image_full_path)
            mask = rebuild_npy(mask_full_path)

            height = img.shape[0]
            width = img.shape[1]
            img_raw = img.flatten().tostring()
            mask_raw = mask.flatten().tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(height),
                                    'width': _int64_feature(width),
                                    'image_raw': _bytes_feature(img_raw),
                                    'mask_raw': _bytes_feature(mask_raw)}
                                    ))
            writer.write(example.SerializeToString())
        writer.close()

    else:
        img_names = get_fnames(PATH_TRAIN_IMG)
        mask_names = get_fnames(PATH_TRAIN_MASK)  

        train_len = int(len(img_names)*0.7)
        train_img_names = img_names[:train_len]
        train_mask_names = mask_names[:train_len]
        val_img_names = img_names[train_len:]
        val_mask_names = mask_names[train_len:]

        for i in range(len(train_img_names)):
            train_image_full_path = PATH_TRAIN_IMG/train_img_names[i]
            train_mask_full_path = PATH_TRAIN_MASK/train_mask_names[i]

            img = rebuild_npy(train_image_full_path)
            mask = rebuild_npy(train_mask_full_path)

            height = img.shape[0]
            width = img.shape[1]
            img_raw = img.flatten().tostring()
            mask_raw = mask.flatten().tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(height),
                                    'width': _int64_feature(width),
                                    'image_raw': _bytes_feature(img_raw),
                                    'mask_raw': _bytes_feature(mask_raw)}
                                    ))

            writer.write(example.SerializeToString())
        writer.close()

        writer = tf.io.TFRecordWriter('./data_project/train/SN_6_val.tfrecords')

        for i in range(len(val_img_names)):
            val_image_full_path = PATH_TRAIN_IMG/val_img_names[i]
            val_mask_full_path = PATH_TRAIN_MASK/val_mask_names[i]

            img = rebuild_npy(val_image_full_path)
            mask = rebuild_npy(val_mask_full_path)

            height = img.shape[0]
            width = img.shape[1]
            img_raw = img.flatten().tostring()
            mask_raw = mask.flatten().tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(height),
                                    'width': _int64_feature(width),
                                    'image_raw': _bytes_feature(img_raw),
                                    'mask_raw': _bytes_feature(mask_raw)}
                                    ))

            writer.write(example.SerializeToString())
        writer.close()

def parse_record(record):
    name_to_features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'mask_raw': tf.io.FixedLenFeature([], tf.string)
    }

    return tf.io.parse_single_example(record, name_to_features)

def decode_record(record):
    parsed_record = parse_record(record)
    image = tf.io.decode_raw(
        parsed_record['image_raw'], out_type=tf.uint8, little_endian=True, fixed_length=None, name=None
    )
    label = tf.io.decode_raw(
        parsed_record['mask_raw'], out_type=tf.uint8, little_endian=True, fixed_length=None, name=None
    )

    height = parsed_record['height']
    width = parsed_record['width']

    image = tf.reshape(image, (height, width, 3))
    label = tf.reshape(label, (height, width))

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    return image, label

def load_dataset(filenames, augment=False):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(
        filenames
    )
    dataset = dataset.with_options(
        ignore_order
    )
    dataset = dataset.map(
        decode_record, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if augment:
        dataset = dataset.map(
            data_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    return dataset

def get_dataset(filenames, batch_size=128, augment=False, epochs=100):
    dataset = load_dataset(filenames, augment=augment)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset