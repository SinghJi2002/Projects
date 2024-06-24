import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import keras

model = load_model("saved_model/")
img_width=150
img_height=150
class_names={0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}

def prediction(img_array):
    cls=model.predict(img_array)
    return(cls)


def main():    
  st.title("Image Upload and Display")
  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    
      # To read file as bytes:
      image = Image.open(uploaded_file).resize((img_width, img_height))
      pic_arr = np.array(image)
      
      # Display the image
      st.image(image, caption='Uploaded Image.', use_column_width=True)
      
      #Image Preprocessing
      img_tensor = pic_arr.reshape(1,150,150,3)
      cls=prediction(img_tensor)
      label = class_names[np.argmax(cls)]
      st.success(label)
      
if __name__ == '__main__':
  main()

    