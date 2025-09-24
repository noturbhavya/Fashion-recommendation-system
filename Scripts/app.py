import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle

# Load ResNet50 without top layer
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

# Add pooling
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Show model summary
#model.summary()

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)   # <-- assign it properly

    return normalized_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

feature_list = []

for file in filenames:
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddings.pkl', 'wb'))
pickle.dump(filenames,open('filenames.pkl', 'wb'))

