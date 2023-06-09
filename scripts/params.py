import os

##################  VARIABLES  ##################
DATA_SIZE=os.environ.get("DATA_SIZE")
LOAD_CHUNK_SIZE=os.environ.get("LOAD_CHUNK_SIZE")
LAND_USE_ARRAY_SIZE=os.environ.get("LAND_USE_ARRAY_SIZE")
TRIAL_SIZE=os.environ.get("TRIAL_SIZE")


MODEL_TARGET=os.environ.get("MODEL_TARGET")

# Your GCP project for GlobalGreening
GCP_PROJECT=os.environ.get("GCP_PROJECT")
GCP_REGION=os.environ.get("GCP_REGION")

# Cloud Storage
BUCKET_NAME=os.environ.get("BUCKET_NAME")

# Compute Engine
INSTANCE=os.environ.get("INSTANCE")

# # Not used yet
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
# MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
# PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
# EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
# GCR_IMAGE = os.environ.get("GCR_IMAGE")
# GCR_REGION = os.environ.get("GCR_REGION")
# GCR_MEMORY = os.environ.get("GCR_MEMORY")

##################  CONSTANTS  #####################

LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Alastair908/GlobalGreening", "raw_data")
LOCAL_OUTPUT_PATH =  os.path.join(os.path.expanduser('~'), "code", "Alastair908/GlobalGreening", "training_outputs")
