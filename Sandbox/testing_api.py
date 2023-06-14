#import streamlit as st
from PIL import Image
import requests 
import os

url = 'http://localhost:8000'

with open(os.path.join('raw_data/demo_pics/','satellite_26.43_45.5_as_bytes.txt'), "w") as f:
    image_bytes = f

#url_path = str(url + "/upload_image", files={'img': image_bytes})
#print(url_path)

print(requests.post(url + "/upload_image", files={'img': image_bytes}))
