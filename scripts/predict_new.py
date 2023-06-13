

#load images into np.array

#check where you have the satellite images - folder + 
image_directory = "bucket/images_for_prediction"   

images_dataset = [] 
image_files = np.sort(os.listdir(image_directory))

# this part of code preprocesses the images

for file in image_files:
    image = Image.open(file)

    if np.asarray(image).shape[2] >3: 
        image = image.convert('RGB')
    image_np = np.asarray(image)
    images_dataset.append(image_np)

images_dataset_np = np.array(images_dataset)
images_dataset_for_pred = images_dataset_np/255.

# set up model 

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

model = build_inception_resnetv2_unet(input_shape = (512, 512, 3))
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=[dice_coef, "accuracy"])

# predictions
model_path = "GlobalGreening/training_outputs/models/20230611-082522_InceptionResNetV2-UNet.h5"
model.load_weights(model_path)

### need to save the model into API memory or cache
# something like this
# app = FastAPI()
# app.state.model = ...

# @app.get("/predict")
# ...

mask_prediction = model.predict(images_dataset_for_pred)
# app.state.model.predict(...)

# prepare single layer for decoding
id2single = {0: 1, 
             1: 1, 
             2: 0, 
             3: 0, 
             4: 0, 
             5: 1, 
             6: 0, 
             7: 0, 
             8: 1, 
             9: 1, 
             10: 1}

def onehot_to_single(onehot, colormap = id2single):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(1,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

pred_single_list = []
for i in range(len(mask_prediction)):
    image_single_layer = onehot_to_single(mask_prediction[i],id2single)
    pred_single_list.append(image_single_layer)

pred_single_layer = np.array(pred_single_list)
