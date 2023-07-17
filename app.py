import streamlit as st
import pandas as pd
import pickle
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import tifffile
import tempfile
import os



def read_tif_image_from_url(url):
    try:
        # Download the image from the URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the image to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(response.content)
            temp_file.close()
            
            # Open the image using tifffile
            image = tifffile.imread(temp_file.name)
            
            
            # Display the image
            tifffile.imshow(image)
            
            # Clean up the temporary file
            os.remove(temp_file.name)
            
            return image
        else:
            print("Error: Failed to download the image. Status code:", response.status_code)
            return None
    except Exception as e:
        print("Error:", str(e))
        return None




# printing the dataset to the web app
st.title("Predict the amount of N, K, and P")
shapefile = gpd.read_file("file/SAMPLE_GESER/Sampel_LSU_DATA_PREDIKSI.shp")
upload = st.text_input("Enter the URL of the Image")




def image_save(result):
    plt.imshow(result, cmap='RdYlGn')
    plt.colorbar(label='N')
    plt.savefig('my_plot.png')



def export_to_grayscale(input_image_path, output_image_path):
    image = Image.open(input_image_path)
    # Convert the image to grayscale
    grayscale_image = image.convert("L")
    # Save the grayscale image
    grayscale_image.save(output_image_path)


if upload!='':
    image = read_tif_image_from_url(upload)
    save_path = "Image.tif"    
    # Save the image to the specified file path
    tifffile.imsave(save_path, image)
    raster = rasterio.open('Image.tif')
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
        export_to_grayscale('my_plot.png', 'out.png')
        st.image('out.png')
        st.write(result)

        
    if PredictN:
        st.subheader("Predicted values of N")
        file = open('model_N.pkl', 'rb')
        model = pickle.load(file)
        predictions = model.predict(df)
        result = predictions.reshape(bands.shape[1:])
        image_save(result)
        export_to_grayscale('my_plot.png', 'out.png')
        st.image('out.png')
        st.write(result)
    if PredictP:
        st.subheader("Predicted values of P")
        file = open('model_P.pkl', 'rb')
        model = pickle.load(file)
        predictions = model.predict(df)
        result = predictions.reshape(bands.shape[1:])
        image_save(result)
        export_to_grayscale('my_plot.png', 'out.png')
        st.image('out.png')
        st.write(result)
        
