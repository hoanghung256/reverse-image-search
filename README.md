# Reverse Image Search using Content-based Image Retrieval (CBIR)

[Data Set](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

-  This is a 242.96 MB of E-commerce product images

## Main Flow

-  Define model: I use the **VGG16**, a **Convolution Neural Network (CNN)** with weights="imagenet" means this VGG16 variant that pre-trained with ImageNet dataset.

-  Preprocessing the image: process the image to match VGG16 input requires.

-  Feature extraction and normalize features: In this project the feature that I extract call **Complex Features** extract from the layer name **fc1** of VGG16.

-  Calculate distance of vector of search image to all vectors of dataset image using **Euclid distance formula**. The smaller the distance, the more similar the image is.

### Getting Started

-  Install Python and all needs libraries.
-  Download data set from link above, unzip and move /dataset folder into project folder.
-  Run the _extract_image_feature.py_ first then the output folder with 2 files _paths.pkl_ and _vectors.pkl_ will be created.
-  Next, run the _search_image_demo.py_ that will give a demo.
