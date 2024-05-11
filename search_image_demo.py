import extract_image_feature as ev

import matplotlib.pyplot as plt
from PIL import Image
import pickle
import numpy as np

# Hard define searching image
search_image = "path/to/your/image.jpg"

# Init model
model = ev.get_extract_model()

# Extract feature for input image
search_vector = ev.extract_vector(model, search_image)

# Load vectors from vectors.pkl
vectors = pickle.load(open("output/vectors.pkl", "rb"))
paths = pickle.load(open("output/paths.pkl", "rb"))

# Calculate distances from search_vector to all vectors
distances = np.linalg.norm(vectors - search_vector, axis=1)

# Sort and get n results with nearest distance
n = 8
indexes = np.argsort(distances)[:n]

results = [(paths[index], distances[index]) for index in indexes]

# Setup plt
axes = []
grid_size = 3
fig = plt.figure(figsize = (10, 5))

# Display searching image first
axes.append(fig.add_subplot(grid_size, grid_size, 1))
axes[-1].set_title("Search Image")
plt.imshow(Image.open(search_image))

# Display results
for i in range(n):
	draw_image = results[i]
	axes.append(fig.add_subplot(grid_size, grid_size, i + 2))
	# Title of results is distance to of each to search image 
	axes[-1].set_title(draw_image[1])
	plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()	
plt.show()

