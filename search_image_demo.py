from io import BytesIO
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import matplotlib.pyplot as plt
from PIL import Image

def url_to_image(url):
	response = requests.get(url)
	buffered = BytesIO(response.content)
	return Image.open(buffered)

# Hard define for test image
test_image = 'test_images/...'

model = SentenceTransformer("clip-ViT-B-32")
client = QdrantClient(url='http://localhost:6333')
search_vector = model.encode(Image.open(test_image))

results = client.search(
	collection_name='souvenirs',
	query_vector=search_vector,
	with_payload=True,
	score_threshold=0.5,
	limit=10
)

# Setup plt
axes = []
grid_size = 3
fig = plt.figure(figsize = (10, 5))

# Display searching image first
axes.append(fig.add_subplot(grid_size, grid_size, 1))
axes[-1].set_title("Search Image")
plt.imshow(Image.open(test_image))

# Display results
for i in range(10):
	draw_image = results[i]
	axes.append(fig.add_subplot(grid_size, grid_size, i + 2))
	# Title of results is distance to of each to search image 
	axes[-1].set_title(draw_image[1])
	plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()	
plt.show()
