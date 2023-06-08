import streamlit as st
import folium
import pandas as pd
from PIL import Image

st.title('GlobalGreening')

file_path='/Users/karakaya/code/Alastair908/GlobalGreening/app_demo/parca_koordinatlari.csv'
# Veri çerçevenizi yükleyin (koordinat bilgilerini içeriyor)
df = pd.read_csv(file_path)

# Haritanın merkez koordinatlarını belirleyin
center = [latitude, longitude]

# Yılları temsil eden liste
years = df["Year"].unique().tolist()

# Slider ile seçilen yılı saklamak için değişken
selected_year = st.slider("Year", min_value=min(years), max_value=max(years))

# Seçilen yıla ait parça koordinatlarını alın
year_df = df[df["Year"] == selected_year]

# Harita oluşturma
m = folium.Map(location=center, zoom_start=12)

# Parça koordinatlarını haritaya ekleyin
for index, row in year_df.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    image_path = row['Image']

    # Resmi yükleyin
    image = Image.open(image_path)

    # Resmi haritaya ekleyin
    folium.raster_layers.ImageOverlay(image_path, bounds=[[lat - 0.001, lon - 0.001], [lat + 0.001, lon + 0.001]], opacity=0.6).add_to(m)

# Haritayı Streamlit uygulamanızda gösterin
st.write(m)
