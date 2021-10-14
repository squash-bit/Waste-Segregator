import os
import sys
import streamlit as st
from get_image import *
from predict import predict_garbage

# design frontend using streamlit
st.sidebar.title("Waste Segregator")

def main():
    st.sidebar.subheader('Load image')
    image_file_uploaded = st.sidebar.file_uploader('Upload an image', type = 'jpg')
    st.sidebar.text('OR')
    image_file_chosen = st.sidebar.selectbox('Select an existing image:', get_list_of_images())

    image_file = []
    if image_file_uploaded:
        image_file = [image_file_uploaded, 0]
    elif image_file_uploaded and image_file_chosen:
        image_file = [image_file_uploaded, 0]
    else:
        image_file = [image_file_chosen, 1]


    if image_file_uploaded and image_file[0] and st.sidebar.button('Load Image'):
        image = get_opened_image(image_file[0])
        st.write("""### Selected Image""", expanded = True)
        st.image(image, use_column_width = True)
        
        # make prediction
        prediction = predict_garbage(image_file)
        st.subheader('Prediction')
        st.markdown(f'The predicted label is: **{prediction}**')

    elif image_file_chosen and image_file[0] and st.sidebar.button('Load'):
        image = get_opened_image(os.path.join(PATH_TO_TEST_IMAGES, image_file[0]))
        st.write("""### Selected Image""", expanded = True)
        st.image(image, use_column_width = True)


        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # make prediction
        prediction = predict_garbage(image_file)
        st.subheader('Prediction')
        st.markdown(f'The predicted label is: **{prediction}**')

if __name__ == '__main__':
    main()
