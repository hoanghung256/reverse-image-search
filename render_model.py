import os

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16

# Get model from lib
vgg16_model = VGG16(weights="imagenet")
vgg16_fc1_layer_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)

# Create model folder if not exist
os.makedirs("model", exist_ok=True)

# Save model into .hdf5 file format
# .hdf5 (or .h5) is file format contains multidimensional arrays of scientific data to store model instance
vgg16_fc1_layer_model.save("model/model.hdf5")

# Check is saved
model = load_model('model/model.hdf5')

print("Convert successfully!!!")
model.summary()
