import numpy as np
from sklearn.cluster import KMeans

# Load similarity matrix
morgan_similarity = np.load('./morgan_similarity_matrix.npy')
distance_matrix = 1 - morgan_similarity

# Perform k-means clustering
n_clusters = 1000
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(distance_matrix)

# Load PDB ID list
with open('./pdbbind_ids.txt', 'r') as f:
    pdb_ids = [line.strip() for line in f]
assert len(pdb_ids) == morgan_similarity.shape[0], "Number of PDB IDs does not match matrix dimensions"

# Create a dictionary to map PDB ID to cluster label
pdb_cluster_map = dict(zip(pdb_ids, cluster_labels))
with open('pdbbind_cluster_kmeans.txt', 'w') as f:
    for pdb_id, cluster in pdb_cluster_map.items():
        f.write(f"{pdb_id}\t{cluster}\n")

# Print statistics for confirmation
cluster_sizes = np.bincount(cluster_labels)
print(f"Number of clusters: {n_clusters}")
print(f"Largest cluster size: {np.max(cluster_sizes)}")
print(f"Smallest cluster size: {np.min(cluster_sizes)}")
print(f"Average cluster size: {np.mean(cluster_sizes):.2f}")