from fastapi import FastAPI
import numpy as np

app=FastAPI()

@app.get('/')

def index():
    return{'ok':True}

@app.get('/predict')
def predict(lat:float,
            lon:float ):


    return np.random.randn(5000,2)


#http://127.0.0.1:8000/predict?latitude=40.7614327§longtitude=-73.9798156&year=1985
