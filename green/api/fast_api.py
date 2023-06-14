import pandas as pd
import numpy as np
import cv2
import io

from fastapi import FastAPI, UploadFile, File
from starlette.responses import Response
#from fastapi.middleware.cors import CORSMiddleware

# from globalgreening.scripts.params import *
# from globalgreening.scripts.data import *
from green.interface.model_load_api import *
# from globalgreening.scripts.preprocessing import *

app = FastAPI()

app.state.model = load_model()


# # Allowing all middleware is optional, but good practice for dev purposes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
# )

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2

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
    #shape = input_for_pred.shape
    
    # do sth to image 
    model = app.state.model
    assert model is not None
     
    pred_mask = model.predict(input_for_pred)
    # shape = pred_mask.shape
    
    # pred_mask_rgp = onehot_to_rgb (pred_mask[0,:,:,:])
    # shape = pred_mask_rgp.shape
    
    onehot = pred_mask[0,:,:,:]
    pred_mask_single = onehot_to_single (onehot)
    #shape = pred_mask_single.shape
    
    # Encoding and responding with the image (prepare response)
    # retval, encoded_imag = cv2.imencode('.png', pred_mask_single) # extension depends on which format is sent from Streamlit
    # im = cv2.imencode('.png', pred_mask_rgp)[1] # extension depends on which format is sent from Streamlit
    # if retval:
    #     success = "True"
    
    # Convert the NumPy array to a Python list
    pred_mask_as_list = pred_mask_single.tolist()
    
    # return { 'return': shape }
    return  { 'predicted_layer': pred_mask_as_list }
    # return json.dumps(pred_mask_single.tolist())
    # return Response(content=encoded_imag.tobytes(), media_type="image/png")


