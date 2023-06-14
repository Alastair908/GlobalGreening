
import numpy as np
import pandas as pd
from PIL import Image
import os
import ipdb
from green.params import *

# key parameters to be used
dataset_root_folder = LOCAL_DATA_PATH
images_dir = 'images_trial_run'
masks_dir = 'masks_trial_run'
load_chunk_size = LOAD_CHUNK_SIZE
trial_size = TRIAL_SIZE 

def get_image_geo_locations(dataset_root_folder, images_dir):
    # generating file names from the directory
    
    images_directory = f'{dataset_root_folder}/{images_dir}'
    print(images_directory)
    image_files = np.sort(os.listdir(images_directory))
    file_names = np.char.rstrip(image_files, '.png')
    file_names = np.char.split(file_names, '_') 

    print(image_files[:3], len(image_files))
    print(file_names[:3], len(file_names))

    # creating df with information about longitude, latitude (can also be used to load the images)
    image_geo_locations = np.zeros((len(file_names),2))
    image_geo_locations = pd.DataFrame(image_geo_locations, columns=['latitude', 'longitude'])

    for image_type in ['latitude', 'longitude']:
        for i in range(len(file_names)):
            file = file_names[i]
            
            if image_type == 'latitude':
                text = file[0]
                image_number = ''.join(num for num in text if num.isdigit())
                latitude = file[1].strip('-') 
    #            print(f'latitude is {latitude}')
                image_geo_locations.at[int(image_number),'latitude'] = latitude
                                    
            elif image_type == 'longitude':
                text = file[0]
                image_number = ''.join(num for num in text if num.isdigit())
                longitude = file[2] 
    #            print(f'longitude is {longitude}')
                image_geo_locations.at[int(image_number),'longitude'] = longitude                     
    return image_geo_locations


def load_images(dataset_root_folder, images_dir):
    
    image_geo_locations = get_image_geo_locations(dataset_root_folder, images_dir)

    # loading images into the list (new code so the files are loaded in correct order (by index))
    images_dataset = []

    for i in range(load_chunk_size):
        image_file =f'image{i}_-{image_geo_locations.iat[i,0]}_{image_geo_locations.iat[i,1]}.png'
        path_image = f'{dataset_root_folder}/{images_dir}/{image_file}'
        print(path_image)
        
        image = Image.open(path_image)
        type(image)
        if np.asarray(image).shape[2] >3: 
            image = image.convert('RGB')
        
        image = np.asarray(image)
        images_dataset.append(image)
        print(f'appended image of size {image.shape}')
    
    images_dataset = np.array(images_dataset)
    
    return images_dataset
   

def load_masks(dataset_root_folder, masks_dir):

    land_use_array_size=LAND_USE_ARRAY_SIZE
    trial_size = TRIAL_SIZE 
    load_range_masks = int(trial_size/land_use_array_size)

    for i in range(load_range_masks):
        mask_file = f'land_use_data_from_{i*land_use_array_size}_to_{(i+1)*land_use_array_size-1}.npy'
        path_mask = f'{dataset_root_folder}/{masks_dir}/{mask_file}'
        print(path_mask)
        
        masks_dataset_dir = f'masks_dataset{i+1}'
        if i = 0:
            masks_dataset = np.load(path_mask)
            print(f'loading array {i+1} into mask dataset with shape {masks_dataset.shape}')
    
        else:
            array_to_append = np.load(path_mask)
            masks_dataset = np.vstack((masks_dataset, array_to_append))
            print(f'appending to masks_dataset an array {i+1} with shape {array_to_append.shape}')
    
    return masks_dataset