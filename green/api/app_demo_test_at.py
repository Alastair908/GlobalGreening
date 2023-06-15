import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
import os
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
import requests
import json

st.set_page_config()


CSS = """
.stApp {
    background-image: url("https://images.unsplash.com/photo-1568832359672-e36cf5d74f54?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8OHx8fGVufDB8fHx8fA%3D%3D&w=1000&q=80");
    background-size: cover;
}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: url("https://images.unsplash.com/photo-1568832359672-e36cf5d74f54?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8OHx8fGVufDB8fHx8fA%3D%3D&w=1000&q=80");
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.write("Yan panel içeriği")

st.markdown("<h1 style='text-align: center; font-family:EB Garamond; color: white; font-size: 120px;'>Global Greening🌿 </h1>", unsafe_allow_html=True)

st.sidebar.success("PROJECT: Global Greening 🤖 ")



def live_demo():
    # Set page tab display
    # st.set_page_config(
    # page_title="Simple Image Uploader",
    # page_icon= '🖼️',
    # layout="wide",
    # initial_sidebar_state="expanded",
    # )
    url = 'http://localhost:8000'
    
    st.header('Location analysis')
    
    ### Create a native Streamlit file upload input
    st.markdown("### Let's do an analysis of greening by location")
    img_file_buffer = st.file_uploader('Upload an image')

    # text_input = st.text_input(
    #     "Input your location to search 👇",
    #     #label_visibility=st.session_state.visibility,
    #     #disabled=st.session_state.disabled,
    #     #placeholder=st.session_state.placeholder,
    # )

#     if text_input:
#         progress_text="Retrieving satellite image.Please wait."
#         my_bar=st.progress(0,text=progress_text)
#         for percent_complete in range(100):
#             time.sleep(0.1)
#             my_bar.progress(percent_complete+1, text=progress_text)
# #    st.image("/Users/karakaya/code/Alastair908/GlobalGreening/app_demo/images/satellite_26.43_45.5.png", text_input)
#             img_file_buffer = st.file_uploader('Upload an image')
    
    if img_file_buffer is not None:
    
      col1, col2 = st.columns(2)
      
      with col1:
        ### Display the image user uploaded
        st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")            
 
      with col2:    
        with st.spinner("Wait for it..."):
        ### Get bytes from the file buffer
          img_bytes = img_file_buffer.getvalue()
        
        
          ### Make request to  API (stream=True to stream response as bytes)
          res = requests.post(url + "/upload_image", files={'img': img_bytes})
        
        
          if res.status_code == 200:      
            ### Display the image returned by the API
            response_data = json.loads(res.content)
            response_array = np.array(response_data['predicted_layer'])
            layers = response_array[:,:,0]
            st.write(layers)
            
          else:
            st.markdown("**Oops**, something went wrong 😓 Please try again.")
            print(res.status_code, res.content)







page_names_to_funcs = {
#    "Plotting Demo": plotting_demo,
    "Live Demo":live_demo
}

demo_name = st.sidebar.selectbox("Choose a demo🌿", list(page_names_to_funcs.keys()))

page_func = page_names_to_funcs[demo_name]
page_func()












