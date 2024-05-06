import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np

# Define model
# VGG16 is a Convolutional Neural Network (CNN)
def get_extract_model():
    # I use weights = "imagenet" means that the model has been pre trained with ImageNet dataset
    vgg16_model = VGG16(weights="imagenet")

    # Fine turning the model
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model

# Preprocessing function: convert image to tensor
def image_preprocess(img):
    # VGG16 model required 244x244 size image and RBG color space
    img = img.resize((224, 224))
    img = img.convert("RGB")

    # Convert img to a numpy array 
    # Each pixel of img in numpy will represented by a 3-dimensional vector of RBGs value: red, green, blue 
    # Size of img is (224, 224, 3) now
    x = image.img_to_array(img)

    # Expand dimension of array from 3 to 4 => create a temp batch (a batch have 4 dimension)
    # Now batch's size is (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # Preprocessing the image with VGG16 requires
    # preprocess_input() provided by Keras
    x = preprocess_input(x)

    return x

# Extract image' feature
def extract_vector(model, image_path):
    print("Processing: ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Feature extraction
    vector = model.predict(img_tensor)[0]

    # Vector normalization by divide to it's normalize vector
    # Normalize vector is size of vector, calculate by np.linalg.norm(vector)
    vector = vector / np.linalg.norm(vector)
    return vector

# Prevent python run this code when import to another file
if __name__ == "__main__":
    dataset_folder = "dataset"
    model = get_extract_model()

    vectors = []
    paths = []
    
    # This is only a demo with 50 first image
    for image_path in os.listdir(dataset_folder)[:50]:
        full_image_path = os.path.join(dataset_folder, image_path)
        image_vector = extract_vector(model, full_image_path)
        vectors.append(image_vector)
        paths.append(full_image_path)

    vector_file = "output/vectors.pkl"
    path_file = "output/paths.pkl"

    pickle.dump(vectors, open(vector_file, "wb"))
    pickle.dump(paths, open(path_file, "wb"))