import zipfile
from datetime import datetime

import requests
import numpy as np
import pandas as pd

# plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import load_batch_of_features_from_store, get_model_predictions, load_model_from_registry 
    

from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout="wide")

# title
# current_date = datetime.strptime('2023-04-03 12:00:00', '%Y-%m-%d %H:%M:%S')
#current_date = pd.to_datetime(datetime.utcnow()).floor('H')
current_date = datetime.strptime('2023-01-05 12:00:00', '%Y-%m-%d %H:%M:%S')
st.title(f'Taxi demand prediction 🚗')
st.header(f'{current_date}')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 7

def load_shape_data_file():

    # download file
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f'{URL} is not available')
    
    # unzip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # load and return shape file
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')

with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = load_shape_data_file()
    st.sidebar.write('✅Shape file was downloaded')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write('✅Inference Features fetched from the store')
    progress_bar.progress(2/N_STEPS)
    print(f'{features}')

with st.spinner(text="Loading ML model from the registry"):
    model = load_model_from_registry()
    st.sidebar.write('✅Model was loaded from the registry')
    progress_bar.progress(3/N_STEPS)

with st.spinner(text="Computing model predictions"):
    results=get_model_predictions(model, features)
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(4/N_STEPS)

