import streamlit as st
import pandas as pd
import numpy as np
import folium
st.header('Global Greening')
import streamlit as st
import folium
from streamlit_folium import folium_static

def main():
    # Colorado'nun enlem ve boylam koordinatları
    colorado_coords = [39.5501, -105.7821]

    # Folium kütüphanesini kullanarak haritayı oluşturun
    m = folium.Map(location=colorado_coords, zoom_start=7)

    # Colorado eyaletini gösteren bir işaretçi ekleyin
    folium.Marker(
        location=colorado_coords,
        popup='Colorado',
        icon=folium.Icon(color='blue', icon='cloud')
    ).add_to(m)

    # Haritayı görüntüleyin
    folium_static(m)



#def main():
    # Yıl seçimi için bir slider ekleme
    #selected_year = st.slider("Yıl Seçin", min_value=2000, max_value=2022)

    # Seçilen yıla ait vegetation indexi resmini yükleme
    #image_path = f"vegetation_index_{selected_year}.png"
    #image = Image.open(image_path)

    # Folium haritasını oluşturma
   # m = folium.Map(location=[39.5501, -105.7821], zoom_start=7)

    # Vegetation indexi resmini haritaya ekleme
    #folium.raster_layers.ImageOverlay(
        #image_path,
        #bounds=[[37.0951, -109.0452], [41.0034, -102.0419]],
        #opacity=0.7
    #).add_to(m)

    # Haritayı görüntüleme
    #folium_static(m)

#if __name__ == "__main__":
    #main()

if __name__ == "__main__":
    main()
st.SessionState
