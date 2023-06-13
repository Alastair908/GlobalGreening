class_tree_cover = '#006400'
class_tree_cover = class_tree_cover.lstrip('#')
class_tree_cover = np.array(tuple(int(class_tree_cover[i:i+2], 16) for i in (0,2,4)))
print(class_tree_cover)

class_shrubland = '#8429F6'
class_land = class_land.lstrip('#')
class_land = np.array(tuple(int(class_land[i:i+2], 16) for i in (0,2,4)))
print(class_land)

class_road = '#6EC1E4'
class_road = class_road.lstrip('#')
class_road = np.array(tuple(int(class_road[i:i+2], 16) for i in (0,2,4)))
print(class_road)

class_vegetation = '#FEDD3A'
class_vegetation = class_vegetation.lstrip('#')
class_vegetation = np.array(tuple(int(class_vegetation[i:i+2], 16) for i in (0,2,4)))
print(class_vegetation)

class_water = '#E2A929'
class_water = class_water.lstrip('#')
class_water = np.array(tuple(int(class_water[i:i+2], 16) for i in (0,2,4)))
print(class_water)

class_unlabeled = '#9B9B9B'
class_unlabeled = class_unlabeled.lstrip('#')
class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2], 16) for i in (0,2,4)))
print(class_unlabeled)


10 #006400 Tree cover
20 #ffbb22 Shrubland
30 #ffff4c Grassland
40 #f096ff Cropland
50 #fa0000 Built-up
60 #b4b4b4 Bare / sparse vegetation
70 #f0f0f0 Snow and ice
80 #0064c8 Permanent water bodies
90 #0096a0 Herbaceous wetland
95 #00cf75 Mangroves
100 #fae6a0 Moss and lichen



labels_dict = {"classes": [
{"title": "Tree cover", "r": 0 , "g": 0 , "b": 0 }, 
{"title": "Shrubland", "r": 132, "g": 41, "b": 246 }, 
{"title": "Grassland", "r": 110, "g": 193, "b": 228 }, 
{"title": "Cropland", "r": 60, "g": 16, "b": 152 }, 
{"title": "Built-up", "r": 254, "g": 221, "b": 58 }, 
{"title": "Bare / sparse vegetation", "r": 155, "g": 155, "b": 155 }
{"title": "Snow and ice", "r": 155, "g": 155, "b": 155 }
{"title": "Permanent water bodies", "r": 155, "g": 155, "b": 155 }
{"title": "Herbaceous wetland", "r": 155, "g": 155, "b": 155 }
{"title": "Mangroves", "r": 155, "g": 155, "b": 155 }
{"title": "Moss and lichen", "r": 155, "g": 155, "b": 155 }
]}

labels_dict_df = pd.DataFrame(labels_dict['classes'])

hex_colors_list = ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000',
                   '#b4b4b4', '#f0f0f0', '#0064c8', '#0096a0', '#00cf75', '#fae6a0']

labels_dict_2 = {'label_10': '#006400', 
                 'label_20': '#ffbb22',
                 'label_30':  '#ffff4c',
                 'label_40': '#f096ff',
                 'label_50' : '#fa0000',
                 'label_60' : '#b4b4b4',
                 'label_70' : '#f0f0f0',
                 'label_80' : '#0064c8',
                 'label_90' : '#0096a0',
                 'label_95' : '#00cf75',
                 'label_100' : '#fae6a0'}

for i in range(len(hex_colors_list)):
    color = hex_colors_list[i].lstrip('#')
    r = int(color[0:2],16)
    g = int(color[2:4],16)
    b = int(color[4:6],16)
    labels_dict_df.at[i,'r'] = r
    labels_dict_df.at[i,'g'] = g
    labels_dict_df.at[i,'b'] = b
    
############
     


VM_masks_dir = 'ESA_worldcover'
VM_output_folder = "jupyter/training_outputs" # tim you need to add it here


# I think this part needs to be different, but you need to create the path_image
image_file =f'image{i}_-{image_geo_locations.iat[i,0]}_{image_geo_locations.iat[i,1]}.png'
path_image = f'{VM_dataset_root_folder}/{VM_images_dir}/{image_file}' 
# after you have path_image, you go from here



#load images into np.array
dataset_folder = "jupyter/raw_data"  # tim you need to add it here
images_dir = "zoomed_photos/zoomed_photos"

images_dataset = []
images_directory = f'{VM_dataset_root_folder}/{images_dir}'
image_files = np.sort(os.listdir(images_directory))

for image_file in image_files:
    image = Image.open(image_file)

    if np.asarray(image).shape[2] >3: 
        image = image.convert('RGB')
    image_np = np.asarray(image)
    images_dataset.append(image_np)

images_dataset_np = np.array(images_dataset)
images_dataset_for_pred = images_dataset_np/255.



# model 

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

K.clear_session()

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

model = build_inception_resnetv2_unet(input_shape = (512, 512, 3))
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=[dice_coef, "accuracy"])

# predictions
model_path = "GlobalGreening/training_outputs/models/20230611-082522_InceptionResNetV2-UNet.h5"
model.load_weights(model_path)
pred_all= model.predict(images_dataset_for_pred)