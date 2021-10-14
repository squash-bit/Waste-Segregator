import os
import sys
from PIL import Image
import streamlit as st
from pathlib import Path

PATH_TO_TEST_IMAGES = './test_images/'      # change status

def get_list_of_images():
    file_list = os.listdir(PATH_TO_TEST_IMAGES)
    return [str(filename) for filename in file_list if str(filename).endswith('.jpg')]

def get_opened_image(image):
    return Image.open(image)
