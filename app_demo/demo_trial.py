import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray

st.set_page_config()

#st.title("Global Greening")

st.markdown("<h1 style='text-align: center; font-family:Courier; color: green; font-size: 120px;'>GLOBAL GREENING🌿 </h1>", unsafe_allow_html=True)

#original_title = '<p style="font-family:Courier; color:Green; font-size: 120px;">GLOBAL GREENING 🌿 </p>'

#st.markdown(original_title, unsafe_allow_html=True)

st.sidebar.success("PROJECT: Global Greening 🤖 ")


import streamlit as st

def intro():
    import streamlit as st

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
    st.write(chart_data)

    MAPBOX_API_KEY = st.secrets["MAPBOX_API_KEY"]

    # AWS Open Data Terrain Tiles
    TERRAIN_IMAGE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

    # Define how to parse elevation tiles
    ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}

    SURFACE_IMAGE = f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}@2x.png?access_token={MAPBOX_API_KEY}"


    ndvi_data=pd.read_csv("/Users/karakaya/code/Alastair908/GlobalGreening/app_demo/csv_largeNDVI.csv")
    # Filter data based on selected year
    #selected_year = st.slider("Choose Year!", min_value=int(ndvi_data['Year'].min()), max_value=int(ndvi_data['Year'].max()), step=5)

    #filtered_data = ndvi_data[ndvi_data['Year'] == str(selected_year)]
    #st.write(Changing NDVI index over time )

    nvdi_positive= ndvi_data.query('NDVI>0')
    nvdi_negative= ndvi_data.query('NDVI<0')
    nvdi_negative['NDVI'] = nvdi_negative['NDVI'].abs()

    initial_view_state=pdk.ViewState(
            latitude=37.42700000000004,
            longitude=-106.39530000000022,
            zoom=10,
            pitch=50,
        )
    column_1=pdk.Layer(
            'ColumnLayer',
            data=nvdi_positive,
            get_position=['longitude', 'latitude'],
            radius=500,
            get_elevation ="NDVI",
            elevation_scale=50,
            get_fill_color=[0, 100, 0, 255],
            pickable=True,
            auto_highlight=True,

)

    column_2=pdk.Layer(
            'ColumnLayer',
            data=nvdi_negative,
            get_position=['longitude', 'latitude'],
            radius=500,
            get_elevation ="NDVI",
            elevation_scale=50,
            get_fill_color=[255, 0, 0, 255],
            pickable=True,
            auto_highlight=True,
    )


    scatter=pdk.Layer(
                'ScatterplotLayer',
                 data=nvdi_positive,
                 get_position=['longitude', 'latitude'],
                 get_color=[0, 42, 115, 55],
                 get_radius=200,
             )
    terrain=pdk.Layer(
     "TerrainLayer", elevation_decoder=ELEVATION_DECODER, texture=SURFACE_IMAGE, elevation_data=TERRAIN_IMAGE)

    r = pdk.Deck(layers=[column_1,column_2,scatter,terrain],

    initial_view_state=initial_view_state)

    st.pydeck_chart(r)


    initial_view_state=pdk.ViewState(
            latitude=37.42700000000004,
            longitude=-106.39530000000022,
            zoom=10,
            pitch=50,
        )
    column=pdk.Layer(
            'ColumnLayer',
            data=nvdi_negative,
            get_position=['longitude', 'latitude'],
            radius=500,
            get_elevation ="NDVI",
            elevation_scale=50,
            get_fill_color=[255, 0, 0, 255],
            pickable=True,
            auto_highlight=True,
)
    scatter=pdk.Layer(
                'ScatterplotLayer',
                 data=nvdi_negative,
                 get_position=['longitude', 'latitude'],
                 get_color=[0, 42, 115, 55],
                 get_radius=200,
             )
    terrain=pdk.Layer(
     "TerrainLayer", elevation_decoder=ELEVATION_DECODER, texture=SURFACE_IMAGE, elevation_data=TERRAIN_IMAGE)

    r = pdk.Deck(layers=[column,scatter,terrain],
    initial_view_state=initial_view_state)

    st.pydeck_chart(r)

    st.write(nvdi_negative['NDVI'])



def charts_demo():
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)

    st.pyplot(fig)



#def team_members():

    #team_members = [
    #{"name": "", "image": "image1.png"},
    #{"name": "Jane Smith", "image": "image2.png"},
    ##{"name": "Michael Johnson", "image": "image3.png"},
    #{"name": "Emily Davis", "image": "image4.png"}
#]

# Yeniden boyutlandırılmış resimlerin boyutu
#image_size = (300, 300)

# Streamlit uygulamasını başlat
#st.title('Takım Üyeleri')

# Dört resmi göstermek için 2x2 bir grid oluşturun
#col1, col2 = st.columns(2)

# Takım üyelerini döngüye alın ve resimleri ve isimleri grid içine yerleştirin
#for i, member in enumerate(team_members):
    #with col1 if i < 2 else col2:
        #image = Image.open(member["image"])
        #resized_image = image.resize(image_size)
        #st.image(resized_image, caption=member["name"], use_column_width=True)

#demo_name = st.sidebar.selectbox("Choose a demo🌿", page_names_to_funcs.keys())
#page_names_to_funcs[demo_name]()

page_names_to_funcs = {
    "Intro": intro,
    "Plotting Demo": plotting_demo,
    "Charts Demo": charts_demo,
    #'team': team
}

#st.sidebar.success("PROJECT: Global Greening 🤖 ")
demo_name = st.sidebar.selectbox("Choose a demo🌿", list(page_names_to_funcs.keys()))

page_func = page_names_to_funcs[demo_name]
page_func()



# Create map
#view_state = pdk.ViewState(latitude=filtered_data['lat'].mean(), longitude=filtered_data['lon'].mean(), zoom=10, pitch=50)
#scatterplot_layer = pdk.Layer('ScatterplotLayer', data=filtered_data, get_position='[lon, lat]', get_color=[0, 255, 0, 160], get_radius=200)
#deck = pdk.Deck(map_style='mapbox://styles/mapbox/satellite-v9', initial_view_state=view_state, layers=[scatterplot_layer])
#st.pydeck_chart(deck)
