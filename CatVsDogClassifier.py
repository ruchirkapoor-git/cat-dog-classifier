import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
from PIL import Image
import os

MODEL_PATH = 'cat_dog_classifier.keras'
train_dir = r'C:\Users\Ruchir Kapoor\OneDrive\Desktop\Data\CatVsDog\training_set'

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        # Data preprocessing and augmentation
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )
        # Build and train model
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, epochs=10, validation_data=val_generator)
        model.save(MODEL_PATH)
    return model

def predict_image(model, img):
    img = img.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

# Streamlit UI
st.title("Cat vs Dog Image Classifier")
st.write("Upload an image of a cat or a dog and the model will predict the class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    model = get_model()
    label = predict_image(model, image_pil)
    st.success(f"Prediction: {label}")
