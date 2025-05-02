import os
import shutil
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

# --- Setup ---
model = FaceAnalysis(name='buffalo_s')
model.prepare(ctx_id=0)  

image_dir = "Images"
output_dir = "people_photos"
os.makedirs(output_dir, exist_ok=True)

all_faces = []

# --- Detect faces ---
for file in os.listdir(image_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {file}. Skipping.")
            continue
        faces = model.get(img)

        for face in faces:
            all_faces.append({
                "embedding": face.embedding,
                "filename": file,
                "bbox": face.bbox
            })

print(f"Total faces found: {len(all_faces)}")

if len(all_faces) == 0:
    print("No faces found. Make sure your Images folder contains valid face images.")
    exit()

# --- Normalize embeddings ---
embeddings = [face["embedding"] for face in all_faces]
embeddings = normalize(embeddings)  # L2 normalize vectors to unit length

# --- Cluster embeddings with DBSCAN ---
clustering = DBSCAN(eps=0.6, min_samples=2, metric='cosine').fit(embeddings)
labels = clustering.labels_

# --- Attach cluster labels back to faces ---
for face, label in zip(all_faces, labels):
    face["cluster"] = label

# --- Organize clusters ---
clusters = {}
for face in all_faces:
    cluster_id = face["cluster"]
    if cluster_id == -1:
        continue  # -1 means noise/unmatched face
    if cluster_id not in clusters:
        clusters[cluster_id] = set()
    clusters[cluster_id].add(face["filename"])

# --- Save full images into cluster folders ---
for cluster_id, filenames in clusters.items():
    person_folder = os.path.join(output_dir, f"person_{cluster_id}")
    os.makedirs(person_folder, exist_ok=True)
    
    for filename in filenames:
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(person_folder, filename)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    print(f"✅ Saved {len(filenames)} photo(s) to {person_folder}")

