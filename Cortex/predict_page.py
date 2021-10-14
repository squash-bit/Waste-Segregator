import streamlit as st
import pickle
import numpy as np

def load_model():
    with open(filename, 'rb') as file:
        saved_model = pickle.load(file)
    return saved_model

loaded_model = load_model()

def predict_garbage(image_name):
    image = Image.open(Path('./' + image_name))

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, loaded_model) + ".")
    return predict_image(example_image, loaded_model)
