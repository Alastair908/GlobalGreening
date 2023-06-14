import numpy as np
import pandas as pd
# from PIL import Image
# import matplotlib.pyplot as plt
# %matplotlib inline
# import os, random, cv2 
# import tensorflow as tf
from google.cloud import storage
import requests
from tensorflow.keras import backend as K
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def load_model():
    model = build_inception_resnetv2_unet(input_shape = (512, 512, 3))
    model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=[dice_coef, "accuracy"])
    
    # option to load from url
    # url = "https://storage.googleapis.com/global-greening/20230611-082522_InceptionResNetV2-UNet.h5"
    # response = requests.get(url)
    # with open("filename.ext", "wb") as file:
    #     model.load_weights(model_path)
    #     file.write(response.content)
    
    #option to load from bucket/blob
    # BUCKET_NAME = "global-greening"
    # storage_filename = "20230611-082522_InceptionResNetV2-UNet.h5"
    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)
    # model_path = bucket.blob(storage_filename)

   #  /Users/Alenka/code/Alastair908/GlobalGreening/green/api/20230611-082522_InceptionResNetV2-UNet.h5
    model_path = "20230611-082522_InceptionResNetV2-UNet.h5"
    # breakpoint()
    model.load_weights(model_path)
    return model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_inception_resnetv2_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained InceptionResNetV2 Model """
    encoder = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = encoder.get_layer("input_1").output           ## (512 x 512)

    s2 = encoder.get_layer("activation").output        ## (255 x 255)
    s2 = ZeroPadding2D(( (1, 0), (1, 0) ))(s2)         ## (256 x 256)

    s3 = encoder.get_layer("activation_3").output      ## (126 x 126)
    s3 = ZeroPadding2D((1, 1))(s3)                     ## (128 x 128)

    s4 = encoder.get_layer("activation_74").output      ## (61 x 61)
    s4 = ZeroPadding2D(( (2, 1),(2, 1) ))(s4)           ## (64 x 64)

    """ Bridge """
    b1 = encoder.get_layer("activation_161").output     ## (30 x 30)
    b1 = ZeroPadding2D((1, 1))(b1)                      ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    dropout = Dropout(0.3)(d4)
    outputs = Conv2D(11, 1, padding="same", activation="softmax")(dropout)

    model = Model(inputs, outputs, name="InceptionResNetV2-UNet")
    return model

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

id2single = {0: 1, 
             1: 0, 
             2: 0, 
             3: 0, 
             4: 0, 
             5: 1, 
             6: 0, 
             7: 0, 
             8: 1, 
             9: 1, 
             10: 1}

def onehot_to_single_new(onehot, colormap = id2single):
    '''Function to decode encoded mask labels into single layer
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to single_id (green or not)
        Output: Decoded array (height x width x 1) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2] )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

def onehot_to_single(onehot, colormap = id2single):
    '''Function to decode encoded mask labels into single layer
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to single_id (green or not)
        Output: Decoded array (height x width x 1) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(1,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)



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

def prepare_label_codes(labels_dict, hex_colors_list):
    labels_dict_df = prepare_labels(labels_dict, hex_colors_list)
    
    label_codes = []
    r= np.asarray(labels_dict_df.r)
    g= np.asarray(labels_dict_df.g)
    b= np.asarray(labels_dict_df.b)

    for i in range(len(labels_dict_df)):
        label_codes.append(tuple([r[i], g[i], b[i]]))
    return label_codes

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