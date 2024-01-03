# Library
import io
import numpy as np
from keras.models import load_model
import tensorflow as tf
import streamlit as st
from streamlit_option_menu import option_menu
from res.src.inference import load
from res.src.inference import prediction
from PIL import Image

st.set_page_config(
    page_title="Banana Leaf Disease Classification App",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="auto",
)

def load_model_resnet():
    model_path = "./app/model/resnet/banana_resnet50_model_10.h5"
    model = load(model_path=model_path)
    return model

def load_model_mobilenet():
    model_path = "./app/model/mobilenetv3/banana_mobile_net_v3_model_10_.h5"
    model = load(model_path=model_path)
    return model

def load_model_efficientnet():
    model_path = "./app/model/efficientnet/banana_efficientnetb4_model_10_.h5"
    model = load(model_path=model_path)
    return model

st.title('Banana Leaf Disease Classification App')

selected = option_menu(None, ['All Models', 'ResNet50', 'MobileNetV3', 'EfficientNetB4'],
                    icons=['globe', 'circle-half', 'circle-half', 'circle-half'], menu_icon='intersect', default_index=0, orientation='horizontal')

if selected == "All Models":

    # Upload image
    uploaded_file = st.file_uploader('Upload a banana leaf picture', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(file_bytes))

        # Convert Image to RGB format (PIL default is RGB)
        pil_image = pil_image.convert('RGB')

        # Displaying image
        st.image(pil_image)

        # Resize image if needed
        pil_image = pil_image.resize((224, 224))

        # Convert image to 4 dimension
        pil_image = np.asarray(pil_image)
        pil_image.shape = (1, 224, 224, 3)

    # Submit button
    submit = st.button('Predict')

    col1, col2, col3 = st.columns(3)

    with col1: 
        with st.container(border=True):
            model = load_model_resnet()
            if submit:
                if uploaded_file is not None:
                    st.header("ResNet50 Predictions")
                    prediction(model, pil_image)
                elif uploaded_file is None:
                    st.error("You have not uploaded any files!")

    with col2:
        with st.container(border=True):
            model = load_model_mobilenet()
            if submit:
                if uploaded_file is not None:
                    st.header("MobileNetV3 Predictions")
                    prediction(model, pil_image)
                elif uploaded_file is None:
                    st.error("You have not uploaded any files!")
        
    with col3: 
        with st.container(border=True):
            model = load_model_efficientnet()
            if submit:
                if uploaded_file is not None:
                    st.header("EfficientNetB4 Predictions")
                    prediction(model, pil_image)
                elif uploaded_file is None:
                    st.error("You have not uploaded any files!")

# RESNET
if selected == "ResNet50":

    # Upload image
    uploaded_file = st.file_uploader('Upload a banana leaf picture', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(file_bytes))

        # Convert Image to RGB format (PIL default is RGB)
        pil_image = pil_image.convert('RGB')

        # Displaying image
        st.image(pil_image)

        # Resize image if needed
        pil_image = pil_image.resize((224, 224))

        # Convert image to 4 dimension
        pil_image = np.asarray(pil_image)
        pil_image.shape = (1, 224, 224, 3)

    # Submit button
    submit = st.button('Predict')
    
    model = load_model_resnet()
    if submit:
        if uploaded_file is not None:
            st.header("ResNet50 Predictions")
            prediction(model, pil_image)
        elif uploaded_file is None:
            st.error("You have not uploaded any files!")

# MOBILENET
if selected == "MobileNetV3":
    # Upload image
    uploaded_file = st.file_uploader('Upload a banana leaf picture', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(file_bytes))

        # Convert Image to RGB format (PIL default is RGB)
        pil_image = pil_image.convert('RGB')

        # Displaying image
        st.image(pil_image)

        # Resize image if needed
        pil_image = pil_image.resize((224, 224))

        # Convert image to 4 dimension
        pil_image = np.asarray(pil_image)
        pil_image.shape = (1, 224, 224, 3)

    # Submit button
    submit = st.button('Predict')
    
    model = load_model_mobilenet()
    if submit:
        if uploaded_file is not None:
            st.header("MobileNetV3 Predictions")
            prediction(model, pil_image)
        elif uploaded_file is None:
            st.error("You have not uploaded any files!")

# EFFICIENT NET
if selected == "EfficientNetB4":

    # Upload image
    uploaded_file = st.file_uploader('Upload a banana leaf picture', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(file_bytes))

        # Convert Image to RGB format (PIL default is RGB)
        pil_image = pil_image.convert('RGB')

        # Displaying image
        st.image(pil_image)

        # Resize image if needed
        pil_image = pil_image.resize((224, 224))

        # Convert image to 4 dimension
        pil_image = np.asarray(pil_image)
        pil_image.shape = (1, 224, 224, 3)

    # Submit button
    submit = st.button('Predict')
    
    model = load_model_efficientnet()
    if submit:
        if uploaded_file is not None:
            st.header("EfficientNetB4 Predictions")
            prediction(model, pil_image)
        elif uploaded_file is None:
            st.error("You have not uploaded any files!")