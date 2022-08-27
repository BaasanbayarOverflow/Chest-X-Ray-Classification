import streamlit as st
import tensorflow as tf
from PIL import Image
from cv2 import resize
from io import BytesIO
from numpy import asarray, expand_dims, argmax

MODEL_WEIGHT_PATH = r'C:\Users\Athenda\Desktop\Python\KDNuggets 20\Pneumonia chest X_Ray\log_weights'


def predict(model, image, size): 
    image = Image.open(BytesIO(image)).convert('RGB')
    image = resize(asarray(image), size)
    image = image.astype('float32') / 255
    image = expand_dims(image, axis=0)

    result = argmax(model.predict(image))
    return result


def buildUI(model_name:str, model_loc:str, size:tuple):
    st.title("Medical Image Classification")
    st.header(f"Detect Pneumonia from chest X-ray Image using {model_name} model")

    try:
        model = tf.keras.models.load_model(MODEL_WEIGHT_PATH + '\\' + model_loc)
    except:
        st.error('Model trained weight file not found')

    uploaded_image = st.file_uploader(label="Upload a chest X-ray Image for Pneumonia detection", 
                                        type=['jpg', 'png', 'jpeg'])
    if (uploaded_image is not None):
        image = uploaded_image.getvalue()
        st.image(image, caption='Uploaded chest X-ray image.', use_column_width=True)

        is_predict = st.button('Predict')
        if (is_predict):
            st.write('Calculating results...')
            result = predict(model, image, size)
            if (result == 0):
                st.success('Healthy lungs.')
            else:
                st.warning('Pneumonia detected.')


def showPageOne():
    st.sidebar.markdown('ConvNN model')
    buildUI('CNN', 'model_conv.h5', (120, 120))


def showPageTwo():
    st.sidebar.markdown('Xception model')
    buildUI('Xception', 'model_opencv.h5', (200, 200))

def createApp():
    page_names_to_function = {
        "CNN": showPageOne,
        "Xception": showPageTwo,
    }

    selected_page = st.sidebar.selectbox('Select an app', page_names_to_function)
    page_names_to_function[selected_page]()


def main():
    createApp()

if __name__ == "__main__":
    main()