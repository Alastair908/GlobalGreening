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
    #st.write(chart_data)

    MAPBOX_API_KEY = st.secrets["MAPBOX_API_KEY"]

    # AWS Open Data Terrain Tiles
    TERRAIN_IMAGE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

    # Define how to parse elevation tiles
    ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}

    SURFACE_IMAGE = f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}@2x.png?access_token={MAPBOX_API_KEY}"


    # Add the 'Year' column
    chart_data['Year'] = ['1985'] * len(data_1985) + ['1990'] * len(data_1990) + ['1995'] * len(data_1995)

    # Filter data based on selected year
    selected_year = st.slider("Choose Year!", min_value=int(chart_data['Year'].min()), max_value=int(chart_data['Year'].max()), step=5)
    filtered_data = chart_data[chart_data['Year'] == str(selected_year)]

    #st.write(Changing NDVI index over time )

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



def team_members():
    
    team_members = [
    {"name": "", "image": "image1.png"},
    {"name": "Jane Smith", "image": "image2.png"},
    {"name": "Michael Johnson", "image": "image3.png"},
    {"name": "Emily Davis", "image": "image4.png"}
]

# Yeniden boyutlandırılmış resimlerin boyutu
image_size = (300, 300)

# Streamlit uygulamasını başlat
st.title('Takım Üyeleri')

# Dört resmi göstermek için 2x2 bir grid oluşturun
col1, col2 = st.columns(2)

# Takım üyelerini döngüye alın ve resimleri ve isimleri grid içine yerleştirin
for i, member in enumerate(team_members):
    with col1 if i < 2 else col2:
        image = Image.open(member["image"])
        resized_image = image.resize(image_size)
        st.image(resized_image, caption=member["name"], use_column_width=True)

demo_name = st.sidebar.selectbox("Choose a demo🌿", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


# Create map
#view_state = pdk.ViewState(latitude=filtered_data['lat'].mean(), longitude=filtered_data['lon'].mean(), zoom=10, pitch=50)
#scatterplot_layer = pdk.Layer('ScatterplotLayer', data=filtered_data, get_position='[lon, lat]', get_color=[0, 255, 0, 160], get_radius=200)
#deck = pdk.Deck(map_style='mapbox://styles/mapbox/satellite-v9', initial_view_state=view_state, layers=[scatterplot_layer])
#st.pydeck_chart(deck)
