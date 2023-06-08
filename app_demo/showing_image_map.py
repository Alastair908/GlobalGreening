import streamlit as st
from PIL import Image
import folium
from streamlit_folium import folium_static
import io
import base64
import os

def main():
    st.title("Uydu Fotoğrafını Haritada Gösterme")

    m = folium.Map(location=[37.0, -109.0], zoom_start=10)

    image_folder = os.path.dirname(__file__)

    # Dosya yolunu oluştururken klasör adını ve dosya adını kullanın
    image_filename = "image.png"
    image_path = os.path.join(image_folder, image_filename)

    # Görüntüyü açma işlemi
    image = Image.open(image_path)

    # Image nesnesini veri URI'sine dönüştürme
    image_data = io.BytesIO()
    image.save(image_data, format='PNG')
    encoded_image = base64.b64encode(image_data.getvalue()).decode()

    # Veri URI'sini kullanarak uydu fotoğrafını haritada gösterme
    folium.raster_layers.ImageOverlay(
        encoded_image,
        [[37.0, -109.0], [37.0, -109.0]],
        opacity=1,
        name='Satellite Image'
    ).add_to(m)

    folium_static(m)

if __name__ == '__main__':
    main()
