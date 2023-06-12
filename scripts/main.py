import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A  # used for augmentation
from IPython.display import SVG
import graphviz
import matplotlib.pyplot as plt
%matplotlib inline
import os, random, cv2 # cv2 used for augmentation
from scripts.params import *
from scripts.data import *
from scripts.model import *
from scripts.preprocessing import *


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


# key parameters to be used
dataset_root_folder = LOCAL_DATA_PATH
images_dir = 'images_trial_run'
masks_dir = 'masks_trial_run'
load_chunk_size = LOAD_CHUNK_SIZE
trial_size = TRIAL_SIZE 

# loading data
images_dataset = load_images(dataset_root_folder, images_dir)
masks_dataset = load_masks(dataset_root_folder, masks_dir)

# Preparing X(images) and y(labels) - to be added to load images later
y = encoding_masks(masks_dataset)
X = preprocess_images (images_dataset)

# Finally we shuffle:
p = np.random.permutation(len(X))
X, y = X[p], y[p]

# first split is for train/val data, second split for test data
split = int(len(X) /6.) 
X_test, X_train = X[:split], X[split:]
y_test, y_train = y[:split], y[split:] 


K.clear_session() 

model = build_inception_resnetv2_unet(input_shape = (512, 512, 3))
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=[dice_coef, "accuracy"])
model.summary()

# setting up the model to run - details for callbacks

def callbacks():
    exponential_decay_fn = exponential_decay(0.0001, 60)

    lr_scheduler = LearningRateScheduler(
        exponential_decay_fn,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath = '/Users/Alenka/code/Alastair908/GlobalGreening/training_outputs/models/First_500_Trial_InceptionResNetV2-UNet.h5',
        save_best_only = True, 
    #     save_weights_only = False,
        monitor = 'val_loss', 
        mode = 'auto', 
        verbose = 1
    )

    earlystop = EarlyStopping(
        monitor = 'val_loss', 
        min_delta = 0.001, 
        patience = 4, 
        mode = 'auto', 
        verbose = 1,
        restore_best_weights = True
    )

    csvlogger = CSVLogger(
        filename= "/Users/Alenka/code/Alastair908/GlobalGreening/training_outputs/metrics/First_500_Trial_model_training.csv",
        separator = ",",
        append = False
    )
    callbacks = [checkpoint, earlystop, csvlogger, lr_scheduler]
    return callbacks

batch_size = 64

steps_per_epoch = np.ceil(float(len(X_train)*0.8) / float(batch_size))
print('steps_per_epoch: ', steps_per_epoch)

validation_steps = np.ceil(float(len(X_train)*0.2) / float(batch_size))
print('validation_steps: ', validation_steps)

history = model.fit(
    X_train, 
    y_train,
    batch_size=batch_size,
    validation_split = 0.2, 
    epochs = 50,
    callbacks=callbacks, 
    verbose=1
)


df_result = pd.DataFrame(history.history)

fig, ax = plt.subplots(1, 4, figsize=(40, 5))
ax = ax.ravel()
metrics = ['Dice Coefficient', 'Accuracy', 'Loss', 'Learning Rate']

for i, met in enumerate(['dice_coef', 'accuracy', 'loss', 'lr']): 
    if met != 'lr':
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('{} vs Epochs'.format(metrics[i]), fontsize=16)
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(metrics[i])
        ax[i].set_xticks(np.arange(0,45,4))
        ax[i].legend(['Train', 'Validation'])
        ax[i].xaxis.grid(True, color = "lightgray", linewidth = "0.8", linestyle = "-")
        ax[i].yaxis.grid(True, color = "lightgray", linewidth = "0.8", linestyle = "-")
    else:
        ax[i].plot(history.history[met])
        ax[i].set_title('{} vs Epochs'.format(metrics[i]), fontsize=16)
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(metrics[i])
        ax[i].set_xticks(np.arange(0,45,4))
        ax[i].xaxis.grid(True, color = "lightgray", linewidth = "0.8", linestyle = "-")
        ax[i].yaxis.grid(True, color = "lightgray", linewidth = "0.8", linestyle = "-")
        
plt.savefig('model_metrics_plot.png', facecolor= 'w',transparent= False, bbox_inches= 'tight', dpi= 150)

model.load_weights("./Trial_InceptionResNetV2-UNet.h5")

count = 0
for i in range(2):
    batch_img,batch_mask = next(testing_gen)
    pred_all= model.predict(batch_img)
    np.shape(pred_all)
    
    for j in range(0,np.shape(pred_all)[0]):
        count += 1
        fig = plt.figure(figsize=(20,8))

        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(batch_img[j])
        ax1.set_title('Input Image', fontdict={'fontsize': 16, 'fontweight': 'medium'})
        ax1.grid(False)

        ax2 = fig.add_subplot(1,3,2)
        ax2.set_title('Ground Truth Mask', fontdict={'fontsize': 16, 'fontweight': 'medium'})
        ax2.imshow(onehot_to_rgb(batch_mask[j],id2code))
        ax2.grid(False)

        ax3 = fig.add_subplot(1,3,3)
        ax3.set_title('Predicted Mask', fontdict={'fontsize': 16, 'fontweight': 'medium'})
        ax3.imshow(onehot_to_rgb(pred_all[j],id2code))
        ax3.grid(False)

        plt.savefig('./predictions/prediction_{}.png'.format(count), facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 200)
        plt.show()

if __name__ == '__main__':
#    preprocess(min_date='2009-01-01', max_date='2015-01-01')
#    train(min_date='2009-01-01', max_date='2015-01-01')
#    evaluate(min_date='2009-01-01', max_date='2015-01-01')
#    pred()