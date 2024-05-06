# Reverse Image Search using Content-based Image Retrieval (CBIR)

[Data Set](https://www.kaggle.com/datasets/theaayushbajaj/cbir-dataset) - This is a 242.96 MB of animal images

### Flow of
    - Define model: I use the **VGG16**, a **Convolution Neural Network (CNN)** with weights="imagenet" means this VGG16 variant has pre-trained with Imagenet dataset.

    - Preprocessing the image: process the image to match VGG16 input requires.

    - Feature extraction and normalize features: In this project the feature that I extract call **Complex features** extract from the layer name **fc1** of VGG16.

    - Calculate distance of vector of search image to all vectors of dataset image using **Euclid distance formula**. The smaller the distance, the more similar the image is.
