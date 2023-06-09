import albumentations as A  # used for augmentation
import cv2 # cv2 used for augmentation
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline

# function to augment
def augment(): #width, height
    transform = A.Compose([
        
# not using crop right now to keep all same structure        
#        A.RandomCrop(width=width, height=height, p=1.0),  
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=[60, 300], p=1.0, interpolation=cv2.INTER_NEAREST),
        A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.3], contrast_limit=0.2, p=1.0),
        A.OneOf([
            A.CLAHE (clip_limit=1.5, tile_grid_size=(8, 8), p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
        ], p=1.0),
    ], p=1.0)
    
    return transform


# visualize the augmentations, this function doesn't work so well 
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 16

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(10, 10)) 

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 12))  

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original Image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original Mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed Image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed Mask', fontsize=fontsize)
        
#    plt.savefig('sample_augmented_image.png', facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 100)

# dictionary with class names + dummy rbg values
labels_dict = {"classes": [
    {"title": "Tree cover", "r": 0 , "g": 0 , "b": 0 }, 
    {"title": "Shrubland", "r": 0, "g": 0, "b": 0 }, 
    {"title": "Grassland", "r": 0, "g": 0, "b": 0 }, 
    {"title": "Cropland", "r": 0, "g": 0, "b": 0 }, 
    {"title": "Built-up", "r": 0, "g": 0, "b": 0 }, 
    {"title": "Bare, sparse vegetation", "r": 0, "g": 0, "b": 0 },
    {"title": "Snow and ice", "r": 0, "g": 0, "b": 0 },
    {"title": "Permanent water bodies", "r": 0, "g": 0, "b": 0 },
    {"title": "Herbaceous wetland", "r": 0, "g": 0, "b": 0 },
    {"title": "Mangroves", "r": 0, "g": 0, "b": 0 },
    {"title": "Moss and lichen", "r": 0, "g": 0, "b": 0 }
    ]}

# loading correct rgb values from hex_color list based on ESA  
hex_colors_list = ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000',
                    '#b4b4b4', '#f0f0f0', '#0064c8', '#0096a0', '#00cf75', '#fae6a0']


# preparing label df with all information: classes, rgb values

def prepare_labels(labels_dict, hex_colors_list):

    labels_dict_df = pd.DataFrame(labels_dict['classes'])
    
    for i in range(len(hex_colors_list)):
        color = hex_colors_list[i].lstrip('#')
        r = int(color[0:2],16)
        g = int(color[2:4],16)
        b = int(color[4:6],16)
        labels_dict_df.at[i,'r'] = r
        labels_dict_df.at[i,'g'] = g
        labels_dict_df.at[i,'b'] = b
    
    return labels_dict_df

# preparing label codes (just rgb values in order of classes)

def prepare_label_codes(labels_dict, hex_colors_list):
    labels_dict_df = prepare_labels(labels_dict, hex_colors_list)
    
    label_codes = []
    r= np.asarray(labels_dict_df.r)
    g= np.asarray(labels_dict_df.g)
    b= np.asarray(labels_dict_df.b)

    for i in range(len(labels_dict_df)):
        label_codes.append(tuple([r[i], g[i], b[i]]))
    return label_codes

# preparing label names (just class names)

def prepare_label_names(labels_dict, hex_colors_list):
    labels_dict_df = prepare_labels(labels_dict, hex_colors_list)   
    label_names= list(labels_dict_df.title)
    return label_names

label_codes = prepare_label_codes(labels_dict, hex_colors_list)
label_names = prepare_label_names(labels_dict, hex_colors_list)

code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}

name2id = {v:k for k,v in enumerate(label_names)}
id2name = {k:v for k,v in enumerate(label_names)}


# one-hot encoder 
def rgb_to_onehot(rgb_mask_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 512 x 512 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    
    # shape prepared for image size and channels = num of classes (instead of 3 RGB colors)
    shape = rgb_mask_image.shape[:2]+(num_classes,)
    
    # encoded_image prepare array with right shaoe 
    encoded_mask = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        # image.reshape flattens and keeps 3 channels, then checks which pixels same as color in colormap
        # then change back to image size for each of 6 channels (based on colormap)
        encoded_mask[:,:,i] = np.all(rgb_mask_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])

    return encoded_mask

def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

def preprocess_images (images):
    images = images/255.
    return images

def encoding_masks(masks_dataset):
    encoded_masks = []

    for i in range(len(masks_dataset)): #len(masks_dataset)
        mask = masks_dataset[i]
        encoded_mask = rgb_to_onehot(mask*255, colormap = id2code)
        encoded_masks.append(encoded_mask)
    
    encoded_masks = np.array(encoded_masks) 
    return encoded_mask