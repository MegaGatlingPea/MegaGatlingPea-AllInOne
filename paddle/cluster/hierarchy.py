import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load similarity matrix
splif_similarity = np.load('./morgan_similarity_matrix.npy')
distance_matrix = 1 - splif_similarity

# Perform hierarchical clustering
print("Performing hierarchical clustering...")
linkage_matrix = linkage(distance_matrix, method='ward')

# Cut the dendrogram to get 1000 clusters
n_clusters = 1000
cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
#threshold = 0.5 
#cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

# Load PDB ID list
with open('./pdbbind_ids.txt', 'r') as f:
    pdb_ids = [line.strip() for line in f]
assert len(pdb_ids) == splif_similarity.shape[0], "Number of PDB IDs does not match matrix dimensions"

# Create a dictionary to map PDB ID to cluster label
pdb_cluster_map = dict(zip(pdb_ids, cluster_labels))
with open('pdbbind_cluster_hierarchy.txt', 'w') as f:
    for pdb_id, cluster in pdb_cluster_map.items():
        f.write(f"{pdb_id}\t{cluster}\n")

# Print some statistics
cluster_sizes = np.bincount(cluster_labels)
print(f"Number of clusters: {n_clusters}")
print(f"Largest cluster size: {np.max(cluster_sizes)}")
print(f"Smallest cluster size: {np.min(cluster_sizes)}")
print(f"Average cluster size: {np.mean(cluster_sizes):.2f}")