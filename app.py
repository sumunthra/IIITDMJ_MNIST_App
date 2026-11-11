
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

st.title("Handwritten Digit Classifier")

@st.cache_resource
def load_model():
    return keras.models.load_model("mnist.keras", compile=False)

model = load_model()

uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
invert = st.checkbox("Black digit on white background", value=True)

if uploaded:
    image = Image.open(uploaded).convert("L")

    if invert:
        image = ImageOps.invert(image)

    image = image.resize((28, 28))
    pixels = np.asarray(image, dtype="float32") / 255.0

    st.image(pixels, caption="Model input", width=150, clamp=True)

    scores = model(pixels[None, ...], training=False).numpy()[0]
    st.success(f"Predicted digit: {int(np.argmax(scores))}")
