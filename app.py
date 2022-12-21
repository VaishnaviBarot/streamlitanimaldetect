import os
import uuid
import flask
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time

def predict(image):
    classifier_model = "Animal1.hdf5"
    IMAGE_SHAPE = (128, 128)
    model = load_model('Animal1.hdf5', compile=False)
    test_image = image.resize((128,128))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'CAMEL',
            'CAT',
            'CATTLE',
            'CHICKEN',
            'CROW',
            'DEER',
            'DOG',
            'EAGLE',
            'ELEPHANT',
            'FOX',
            'GOAT',
            'HORSE',
            'LEOPARD',
            'LION',
            'MONKEY',
            'PARROT',
            'PEACOCK',
            'PIG',
            'RABBIT',
            'SHEEP',
            'SPARROW',
            'TIGER',]
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'CAMEL':0,
            'CAT':0,
            'CATTLE':0,
            'CHICKEN':0,
            'CROW':0,
            'DEER':0,
            'DOG':0,
            'EAGLE':0,
            'ELEPHANT':0,
            'FOX':0,
            'GOAT':0,
            'HORSE':0,
            'LEOPARD':0,
            'LION':0,
            'MONKEY':0,
            'PARROT':0,
            'PEACOCK':0,
            'PIG':0,
            'RABBIT':0,
            'SHEEP':0,
            'SPARROW':0,
            'TIGER':0,
        }

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result



def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)

if __name__ == "__main__":
    main()
