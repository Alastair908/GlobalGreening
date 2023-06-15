import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File
#from fastapi.middleware.cors import CORSMiddleware
from green.interface.model_load_api import *

app = FastAPI()

app.state.model = load_model()

# # Allowing all middleware is optional, but good practice for dev purposes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
# )

@app.get("/")
def root():
    return {
    'greeting': 'Hello'
    }

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    image_as_array = np.asarray(image)

    image_preproc = image_as_array/255.
    input_for_pred = image_preproc.reshape(1,512,512,3)
    
    # predict the mask
    model = app.state.model
    assert model is not None
     
    pred_mask = model.predict(input_for_pred)  
    onehot = pred_mask[0,:,:,:]
    pred_mask_single = onehot_to_single (onehot)
    
    
    # Convert the NumPy array to a Python list
    pred_mask_as_list = pred_mask_single.tolist()
    
    return  { 'predicted_layer': pred_mask_as_list }