import numpy as np  
import os
import ipdb
from green.params import *
from scripts import data

dataset_root_folder = LOCAL_DATA_PATH
dataset_name = 'images_trial_run'
mask_set_name = 'masks_trial_run'

def check_image_dataset():
    images_dataset = data.load_images(dataset_root_folder, dataset_name)
    len(images_dataset) = DATA_SIZE
    assert images_dataset[0].shape == (512,512,3)
    assert type(images_dataset[0]) == np.ndarray
    type(images_dataset) = np.ndarray
    
