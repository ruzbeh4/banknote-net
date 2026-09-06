import pickle
import json
import numpy as np

# Load your pickle file
with open('./src/trained_models/cosine_centroids.pkl', 'rb') as f:
    centroids = pickle.load(f)

# Convert numpy arrays to standard Python lists
centroids_json = {str(k): v.tolist() for k, v in centroids.items()}

# Save it as a JSON file
with open('./src/trained_models/cosine_centroids.json', 'w') as f:
    json.dump(centroids_json, f)

print("Successfully converted to centroids.json!")