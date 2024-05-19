import os
import urllib
from typing import Optional
from sentence_transformers import SentenceTransformer

import pandas as pd
from PIL import Image
from qdrant_client import QdrantClient, models
import uuid

# Get model
model = SentenceTransformer('clip-ViT-B-32')

# Create Qdrant collection
client = QdrantClient(url='http://localhost:6333')
client.get_collections()

client.recreate_collection(
    collection_name='souvenirs',
    vectors_config=models.VectorParams(
        size=512,
        # Using the Cosine distance formula to calculate the distance between vectors
        distance=models.Distance.COSINE
    )
)

client.get_collection('souvenirs')

accept_extensions = ['.jpg', '.jpeg', '.png']
# Download image by url
def download_image(url: str) -> Optional[str]:
	basename = os.path.basename(url)
	filename, file_extension = os.path.splitext(basename)

	if file_extension.lower() not in accept_extensions:
		return None

	print(f'Downloading {basename}...')
	target_path = f'dataset/souvenirs/{basename}'

	if not os.path.exists(target_path):
		try:
			# Download image by url and save as target_path
			urllib.request.urlretrieve(url, target_path)
			print(f'Download successfully {url}')
		except:
			print(f'Error download url: {url}')
			return None

	return target_path

def is_path_exists(path):
	return os.path.exists(path) if pd.notnull(path) else False

data_frame = pd.read_csv('dataset/qualuuniem.csv')

# Download images by mapping with image column in data_frame
# then assign into LocalImagePath column
data_frame['LocalImagePath'] = data_frame['image'].map(download_image)

images = data_frame['LocalImagePath'].map(Image.open).toList()

# Extract vector from image
vectors = model.encode(images, show_progress_bar=True)

data_frame['vector'] = vectors.tolist()

# Insert downloaded images into Qdrant database
total_image = len(data_frame)
for i, row in data_frame.itterrows():
	client.upsert(
		collection_name='souvenirs',
		points=[
			models.PointStruct(
				id=uuid.uuid4().hex,
				vector=row['vector'],
				payload={
					'Name': row['name'],
					'Image': row['image'],
					'Link': row['link']
				}
			)
		]
	)
	print(f'Inserted the {i}th/{total_image}')

