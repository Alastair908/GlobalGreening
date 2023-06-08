import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A  # used for augmentation
from IPython.display import SVG
import graphviz
import matplotlib.pyplot as plt
%matplotlib inline
import os, random, cv2 # cv2 used for augmentation

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import optimizers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input

# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import model_to_dot, plot_model, image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout

from sklearn.preprocessing import MinMaxScaler





if __name__ == '__main__':
#    preprocess(min_date='2009-01-01', max_date='2015-01-01')
#    train(min_date='2009-01-01', max_date='2015-01-01')
#    evaluate(min_date='2009-01-01', max_date='2015-01-01')
#    pred()