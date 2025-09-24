import pickle
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

# Load embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

# Add pooling
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Load and preprocess test image
img = image.load_img('sample/tshirt.webp', target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Extract features
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Find nearest neighbors
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

# Print recommended image filenames
print("Recommended images:")
for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img, (512,512)))
    cv2.waitKey(0)

