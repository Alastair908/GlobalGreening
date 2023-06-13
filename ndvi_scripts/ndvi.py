import pandas as pd
import numpy as np
from PIL import Image

BUCKET_NAME = "GlobalGreening"
images_dir = "NDVI_2002" # NDVI_2010 NDVI_2014 NDVI_2018 NDVI_2022
masks_dir = "ESA_worldcover"
dataset_root_folder = "/Users/timfrith/code/Alastair908/GlobalGreening/raw_data"

def load_images_ndvi(dataset_root_folder, images_dir, DataFrame_locations):
    images_dataset = []

    for index,row in DataFrame_locations.iterrows():
        id_ = index
        longitude = row['longitude']
        latitude = row['latitude']

        image_file =f'image{id_}_{longitude}_{latitude}.png'
        path_image = f'{dataset_root_folder}/{images_dir}/{image_file}'
        # print(f'loading the image from these file {path_image}')

        image = Image.open(path_image)
        type(image)
        if np.asarray(image).shape  == (512,512,2):
            image = image.convert('L')

        image = np.asarray(image)
        images_dataset.append(image)
        # print(f'appended image of size {image.shape}')

    images_dataset = np.array(images_dataset)

    return images_dataset

def ndvi_mean(mask, image):
    '''Calculate the mean result of the mask and image'''

    np_image = np.array(image)

    point = mask[:,:,0] * np_image

    mean =  point[np.nonzero(point)].mean()

    return mean
