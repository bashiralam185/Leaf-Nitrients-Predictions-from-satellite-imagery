import streamlit as st
import pandas as pd
import pickle
import geopandas as gpd
import rasterio
import numpy as np




# printing the dataset to the web app
st.title("Predict the amount of N, K, and P")
# uploaded_file = st.file_uploader("Upload your TIF image here...")

shapefile = gpd.read_file("file/SAMPLE_GESER/Sampel_LSU_DATA_PREDIKSI.shp")
upload = st.text_input("Enter the path of the image")

if upload!='':
    raster = rasterio.open(upload)
    bands = raster.read()
    red_band = bands[0].astype(float)
    nir_band = bands[3].astype(float)
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
    ndvi_band = np.expand_dims(ndvi, axis=0)
    dataset = np.concatenate((bands, ndvi_band), axis=0)
    raster_transform = raster.transform

    df = pd.DataFrame(columns=['band1', 'band2', 'band3', 'band4'])
    for index, row in shapefile.iterrows():
        
            row, col = rasterio.transform.rowcol(raster_transform, row['geometry'].x, row['geometry'].y)

            # Get the band values at the pixel
            band1 = dataset[0, row, col]
            band2 = dataset[1, row, col]
            band3 = dataset[2, row, col]
            band4 = dataset[3, row, col]
            ndvi = dataset[4, row, col]

            # Add the coordinates and the band values to the dataframe
            df = df.append({'band1': band1, 'band2': band2, 'band3': band3, 'band4': band4, 'ndvi': ndvi}, ignore_index=True)
    Predict = st.button("Predict K")
    PredictN = st.button("Predict N")
    PredictP = st.button("Predict P")


    if Predict:
        st.subheader("Predicted values of K")
        file = open('model_K.pkl', 'rb')
        model = pickle.load(file)
        predictions = model.predict(df)
        st.write(predictions)
    if PredictN:
        st.subheader("Predicted values of N")
        file = open('model_N.pkl', 'rb')
        model = pickle.load(file)
        predictions = model.predict(df)
        st.write(predictions)
    if PredictP:
        st.subheader("Predicted values of P")
        file = open('model_P.pkl', 'rb')
        model = pickle.load(file)
        predictions = model.predict(df)
        st.write(predictions)
        
