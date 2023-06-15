FROM python:3.10.6
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY green green
COPY setup.py setup.py
RUN pip install .

CMD uvicorn green.api.fast_api:app --host 0.0.0.0 --port $PORT

