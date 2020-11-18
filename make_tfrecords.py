import numpy as np
import tensorflow as tf
from utils.datagen import construct_records

construct_records('./data_project/test/SN_6_test.tfrecords', test=True)