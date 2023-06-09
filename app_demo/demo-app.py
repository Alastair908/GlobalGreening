
#import pygmt
# Create a new PyGMT Figure
#fig = pygmt.Figure()
# Define the region of interest on the map
# (in this case, the region covering Colorado)
#fig.basemap(region=[-109.045223, -102.041524, 36.993076, 41.003444], projection="M6i", frame=True)
# Sample coordinates to plot (longitude, latitude)
#coordinates = [[ 39.7392,-104.9903]]  # Coordinates for Denver, Colorado
# Plot the data onto the map
#fig.plot(data=coordinates, style="c0.2c", color="blue", pen="black")
# Display the map
#fig.show()

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import os
MAPBOX_API_KEY = st.secrets.mapbox.key

# AWS Open Data Terrain Tiles
TERRAIN_IMAGE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# Define how to parse elevation tiles
ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}

SURFACE_IMAGE = f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}@2x.png?access_token={MAPBOX_API_KEY}"

chart_data = pd.DataFrame(
   np.random.randn(15000, 2) / [10, 10] + [39.7302, -104.9903],
   columns=['lat', 'lon'])


st.title('GlobalGreening')
st.text(chart_data.shape)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=39.7302,
        longitude=-104.9903,
        zoom=10,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=chart_data,
           get_position='[lon, lat]',
           radius=300,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
           get_color='[0, 255, 0, 160]'
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='[0, 255, 0, 160]',
            get_radius=200,
        ),
        pdk.Layer(
    "TerrainLayer", elevation_decoder=ELEVATION_DECODER, texture=SURFACE_IMAGE, elevation_data=TERRAIN_IMAGE
)
    ],
))
#Adding bar graph

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"])

st.bar_chart(chart_data)


#Adding text box

sentence = st.text_input('Input your sentence here:')

if sentence:
    st.write(my_model.predict(sentence))


#Adding background images
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()



#MAPBOX_API_KEY = 'sk.eyJ1IjoibnVya2FybGlkYWc5MyIsImEiOiJjbGluOGZtNm0wdnNyM3Fwa3BvMDBvb2c0In0.PAXTx-M3mD_5IdSduukIgQ'

# AWS Open Data Terrain Tiles
#TERRAIN_IMAGE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# Define how to parse elevation tiles
#ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}

#SURFACE_IMAGE = f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}@2x.png?access_token={MAPBOX_API_KEY}"

#terrain_layer = pdk.Layer(
#    "TerrainLayer", elevation_decoder=ELEVATION_DECODER, texture=SURFACE_IMAGE, elevation_data=TERRAIN_IMAGE
#)

#view_state = pdk.ViewState(latitude=39.24, longitude=-104.18, zoom=11.5, bearing=10, pitch=60)

#r = pdk.Deck(terrain_layer, initial_view_state=view_state)
