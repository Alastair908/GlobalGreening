import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
#from app_demo.api.fast import predict
import requests
import os
import base64

st.set_page_config()

@st.cache_data()

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('../output.png')


#st.title("Global Greening")

#st.markdown("<h1 style='text-align: center; font-family:Courier; color: green; font-size: 120px;'>GLOBAL GREENING🌿 </h1>", unsafe_allow_html=True)

#original_title = '<p style="font-family:Courier; color:Green; font-size: 120px;">GLOBAL GREENING 🌿 </p>'

#st.markdown(original_title, unsafe_allow_html=True)


#st.sidebar.success("PROJECT: Global Greening 🤖 ")

lon = st.number_input('Enter lon:')
lat = st.number_input('Enter lat:')
# url = 'http://localhost:8080/predict'

# params = {
#     'lon':10.0, # 0 for Sunday, 1 for Monday, ...
#     'lat': 20.2
# }

# response = requests.get(url, params=params)
# res=response.json() #=> {wait: 64}


#conn = st.experimental_connection('gcs', type=FilesConnection)
#image_test = conn.read("gs://global-greening/zoomed_photos/image10042_-104.47_40.33.png", type="png", ttl=600)
#ttl is for timeout by seconds
#input_img = st.image('https://storage.cloud.google.com/global-greening/zoomed_photos/image10042_-104.47_40.33.png')


#Preprocess
#load images into np.array
#dataset_folder = "jupyter/raw_data"  # tim you need to add it here
#images_dir = "zoomed_photos/zoomed_photos"

# images_dataset = []
# VM_dataset_root_folder = 'https://storage.cloud.google.com/global-greening'
# images_dir = 'zoomed_photos'
# images_directory = f'{VM_dataset_root_folder}/{images_dir}'
# image_files = np.sort(os.listdir(images_directory))

# for image_file in image_files:
#     image = Image.open(input_img)

#     if np.asarray(image).shape[2] >3:
#         image = image.convert('RGB')
#     image_np = np.asarray(image)
#     images_dataset.append(image_np)

# images_dataset_np = np.array(images_dataset)
# images_dataset_for_pred = images_dataset_np/255.


# Model weights here https://storage.cloud.google.com/global-greening/models

# Create Model

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


# Load the weights
model.load_weights('https://storage.cloud.google.com/global-greening/models')
# Model predict mask by inputing images_dataset_for_pred
model.predict(images_dataset_for_pred)
#ask alastair for code to convert to one single value per pic

# convert this list of value into dataframe chart_data --- show on graph

def intro():

    st.markdown(res)
    st.markdown("<h1 style='text-align: center; font-family:Courier; color: green; font-size: 120px;'>GLOBAL GREENING🌿 </h1>", unsafe_allow_html=True)
    st.write("# Welcome to Streamlit! 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )


def plotting_demo():



    # Generate random data for each year
    data_1985 = np.random.randn(5000,2) + [39.7302, -104.9903]
    data_1990 = np.random.randn(5000,2) / [5, 5] + [39.7302, -104.9903]
    data_1995 = np.random.randn(5000,2) / [20, 20] + [39.7302, -104.9903]


    # Combine the data for each year into a single DataFrame
    chart_data = pd.DataFrame(
        np.concatenate([data_1985, data_1990, data_1995],axis=0),
        columns=['lat', 'lon']
    )
    #st.write(data_1985[1])
    #st.write(data_1985[2])
    #st.write(chart_data)

    MAPBOX=st.secrets["MAPBOX_API_KEY"]

    # AWS Open Data Terrain Tiles
    TERRAIN_IMAGE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

    # Define how to parse elevation tiles
    ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}

    SURFACE_IMAGE = f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}@2x.png?access_token={MAPBOX}"


    # Add the 'Year' column
    chart_data['Year'] = ['1985'] * len(data_1985) + ['1990'] * len(data_1990) + ['1995'] * len(data_1995)

    #st.markdown(chart_data)

    # Filter data based on selected year
    selected_year = st.slider("Choose Year!", min_value=int(chart_data['Year'].min()), max_value=int(chart_data['Year'].max()), step=5)
    filtered_data = chart_data[chart_data['Year'] == str(selected_year)]

    #st.write(Changing NDVI index over time )

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=39.7302,
            longitude=-104.9903,
            zoom=7,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
            'ColumnLayer',
            data=chart_data,
            get_position='[lon, lat]',
            radius=300,
            # get_elevatiıon = ???
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
            get_color='[0, 255, 240 160]'
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=chart_data,
                get_position='[lon, lat]',
                get_color='[0, 42, 115, 55]',
                get_radius=200,
            ),
            pdk.Layer(
        "TerrainLayer", elevation_decoder=ELEVATION_DECODER, texture=SURFACE_IMAGE, elevation_data=TERRAIN_IMAGE
    )
        ],
    ))


def charts_demo():
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)

    st.pyplot(fig)


#def team():
   # team_members = [
        #{"name": "", "image": "/Users/karakaya/code/Alastair908/GlobalGreening/output.png"},
        #{"name": "Jane Smith", "image": "/Users/karakaya/code/Alastair908/GlobalGreening/output.png"},
        #{"name": "Michael Johnson", "image": "/Users/karakaya/code/Alastair908/GlobalGreening/output.png"},
        #{"name": "Emily Davis", "image": "/Users/karakaya/code/Alastair908/GlobalGreening/output.png"}
    #]

    # Yeniden boyutlandırılmış resimlerin boyutu
    #image_size = (300, 300)

    # Streamlit uygulamasını başlat
    #st.title('Meet Our Team!')

    # Dört resmi göstermek için 2x2 bir grid oluşturun
    #col1, col2 = st.columns(2)

    # Takım üyelerini döngüye alın ve resimleri ve isimleri grid içine yerleştirin
    #for i, member in enumerate(team_members):
        #with col1 if i < 2 else col2:
            #image = Image.open(member["image"])
            #resized_image = image.resize(image_size)
            #st.image(resized_image, caption=member["name"], use_column_width=True)



page_names_to_funcs = {
    "Intro": intro,
    "Plotting Demo": plotting_demo,
    "Charts Demo": charts_demo,
    #'team': team
}

st.sidebar.success("PROJECT: Global Greening 🤖 ")
demo_name = st.sidebar.selectbox("Choose a demo🌿", list(page_names_to_funcs.keys()))

page_func = page_names_to_funcs[demo_name]
page_func()


# Create map
#view_state = pdk.ViewState(latitude=filtered_data['lat'].mean(), longitude=filtered_data['lon'].mean(), zoom=10, pitch=50)
#scatterplot_layer = pdk.Layer('ScatterplotLayer', data=filtered_data, get_position='[lon, lat]', get_color=[0, 255, 0, 160], get_radius=200)
#deck = pdk.Deck(map_style='mapbox://styles/mapbox/satellite-v9', initial_view_state=view_state, layers=[scatterplot_layer])
#st.pydeck_chart(deck)
