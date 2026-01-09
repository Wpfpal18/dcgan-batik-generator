import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

IMG_SIZE = 64
LATENT_DIM = 128
MODEL_PATH = os.path.join("output", "generator_final.keras")

st.set_page_config(page_title="DCGAN Batik Generator", layout="centered")

st.title("Indonesian Batik Generator")
st.write("DCGAN Image Generator untuk Motif Batik Indonesia")

@st.cache_resource(show_spinner=True)
def load_generator():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model belum ditemukan. Jalankan training terlebih dahulu")
    st.stop()

generator = load_generator()

st.subheader("Generate Batik")

num_images = st.slider("Jumlah gambar", 1, 9, 1)

if st.button("Generate"):
    noise = tf.random.normal([num_images, LATENT_DIM])
    imgs = generator(noise, training=False)
    imgs = (imgs * 127.5 + 127.5).numpy().astype(np.uint8)

    cols = st.columns(min(3, num_images))
    for i in range(num_images):
        img = Image.fromarray(imgs[i])
        cols[i % len(cols)].image(img, width=200)

st.subheader("Contoh Hasil Training")

sample_files = sorted(
    [f for f in os.listdir("output") if f.startswith("sample_")]
)

if len(sample_files) > 0:
    selected = st.selectbox("Pilih sample", sample_files)
    img = Image.open(os.path.join("output", selected))
    st.image(img, caption=selected, width=300)
else:
    st.write("Belum ada sample image")
