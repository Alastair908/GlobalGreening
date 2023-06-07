import streamlit as st
import folium
from PIL import Image

# Görüntülerin bulunduğu dizin
image_dir = "path_to_image_directory"

# Haritanın merkez koordinatlarını belirleyin
center = [latitude, longitude]

# Yılları temsil eden liste
years = [1985, 1986, ..., 2022]

# Slider ile seçilen yılı saklamak için değişken
selected_year = st.slider("Yıl Seçin", min_value=min(years), max_value=max(years))

# Seçilen yılın görüntüsünü yükleme
image_path = f"{image_dir}/{selected_year}.png"
image = Image.open(image_path)

# Seçilen yılın resmini gösterme
st.image(image, caption=str(selected_year), use_column_width=True)

# Harita oluşturma
m = folium.Map(location=center, zoom_start=12)

# Resmi haritaya ekleyin
folium.raster_layers.ImageOverlay(image_path, bounds=[[min_lat, min_lon], [max_lat, max_lon]], opacity=0.6).add_to(m)

# Haritayı Streamlit uygulamanızda gösterin
folium_static(m)


----------------------------------------------------

import streamlit as st
import folium
import pandas as pd

# Veri çerçevenizi yükleyin
df = pd.read_csv(".csv")

# Haritanın merkez koordinatlarını belirleyin
center = [latitude, longitude]

# Harita oluşturma
m = folium.Map(location=center, zoom_start=12)

# Veri çerçevesindeki her bir satır için işlemler yapın
for index, row in df.iterrows():
    # Resim parçasının merkez koordinatlarını alın
    lat = row['latitude']
    lon = row['longitude']

    # Markörü haritaya ekleyin
    folium.Marker(location=[lat, lon]).add_to(m)

# Haritayı Streamlit uygulamanızda gösterin
st.write(m)
