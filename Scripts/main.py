import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

st.title('👗 Fashion Recommender System')

# File upload function
def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error: {e}")
        return 0

# Feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommendation
def recommend(features, feature_list, n_neighbors):
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Upload and process image
uploaded_file = st.file_uploader("Choose an Image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):

        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image")

        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # 🔥 Slider to control number of recommendations
        num_recommendations = st.slider("Number of Recommendations", min_value=5, max_value=20, step=5, value=10)

        indices = recommend(features, feature_list, n_neighbors=num_recommendations+1)

        st.subheader(f"Top {num_recommendations} Recommendations")

        cols = st.columns(5)  # show 5 images per row

        for i in range(1, num_recommendations+1):  # skip the first because it's the same image
            with cols[i % 5]:
                st.image(filenames[indices[0][i]], caption=f"Suggestion {i}")

    else:
        st.header("Some error occurred in file upload")
