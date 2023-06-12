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

    #st.write(chart_data)

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
