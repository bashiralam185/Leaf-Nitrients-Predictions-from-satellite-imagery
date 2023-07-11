import streamlit as st
import pandas as pd
import pickle
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt




# printing the dataset to the web app
st.title("Predict the amount of N, K, and P")
# uploaded_file = st.file_uploader("Upload your TIF image here...")

shapefile = gpd.read_file("file/SAMPLE_GESER/Sampel_LSU_DATA_PREDIKSI.shp")
upload = st.text_input("Enter the path of the image")

def image_save(result):
    plt.imshow(result, cmap='RdYlGn')
    plt.colorbar(label='N')
    plt.savefig('my_plot.png')


if upload!='':
    raster = rasterio.open(upload)
    bands = raster.read()
    red_band = bands[0].astype(float)
    nir_band = bands[3].astype(float)
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
    ndvi_band = np.expand_dims(ndvi, axis=0)
    dataset = np.concatenate((bands, ndvi_band), axis=0)
    df = np.transpose(dataset, (1, 2, 0)).reshape(-1, dataset.shape[0])
    
    Predict = st.button("Predict K")
    PredictN = st.button("Predict N")
    PredictP = st.button("Predict P")

    if Predict:
        st.subheader("Predicted values of K")
        file = open('model_K.pkl', 'rb')
        model = pickle.load(file)
        predictions = model.predict(df)
        result = predictions.reshape(bands.shape[1:])
        image_save(result)
        st.image('my_plot.png')
        st.write(result)
        
    if PredictN:
        st.subheader("Predicted values of N")
        file = open('model_N.pkl', 'rb')
        model = pickle.load(file)
        predictions = model.predict(df)
        result = predictions.reshape(bands.shape[1:])
        image_save(result)
        st.image('my_plot.png')
        st.write(result)
    if PredictP:
        st.subheader("Predicted values of P")
        file = open('model_P.pkl', 'rb')
        model = pickle.load(file)
        predictions = model.predict(df)
        result = predictions.reshape(bands.shape[1:])
        image_save(result)
        st.image('my_plot.png')
        st.write(result)
        
