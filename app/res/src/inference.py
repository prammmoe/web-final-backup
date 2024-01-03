import numpy as np
from keras.models import load_model
import tensorflow as tf
import streamlit as st

def load(model_path: str):
    model = load_model(model_path)
    return model

def prediction(model, pil_image):
    CLASS_NAMES = ['Pestalotiopsis', 'Cordana', 'Healthy', 'Sigatoka']
    with st.status("Predicting data...", expanded=True) as status:
        pred = model.predict(pil_image)
        result = CLASS_NAMES[np.argmax(pred)]
        tempdict = {}
        for idx, value in enumerate(pred[0]):
            tempdict[CLASS_NAMES[idx]] = value
        st.write(tempdict)

        st.subheader('This is a ' + result + ' banana leaf.')
        status.update(label="Prediction complete!", state="complete", expanded=False)
