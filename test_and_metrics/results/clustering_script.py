from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def load_matrix(path):
    return np.loadtxt(path, delimiter="\t")

def to_distance_matrix(matrix):
    distance_matrix = 1.0 - matrix
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

def choose_best_k(distance_matrix, Z, k_min=2, k_max=5):
    scores = {}
    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")

        if len(set(labels)) < 2:
            continue
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        scores[k] = score
 
    best_k = max(scores, key=scores.get)
    return best_k, scores

def cluster_similarity_matrix(matrix, k = None,k_min=2, k_max=5):
    """
    If k is None, automatically chooses the best k between k_min and k_max using silhouette score. 
    If k specified, uses that k for clustering, exacly k cluisters. 
    """
    distance_matrix = to_distance_matrix(matrix)
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")

    #Plot the dendrogram 
    plt.figure(figsize=(10,6))
    d_ward = dendrogram(Z, leaf_rotation=90, leaf_font_size=10, link_color_func=lambda x: 'black')
    plt.title('Linkage Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Cluster Distance')
    plt.show()
 
    if k is None:
        k, scores = choose_best_k(distance_matrix, Z, k_min, k_max)
        print("Silhouette score:")    
        for kk, s in scores.items():
            marker = "  <-- best one" if kk == k else ""
            print(f"  k={kk}: {s:.4f}{marker}")
    
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, k

def cluster_stats(matrix, labels):
    clusters = {}
    n = len(labels)

    for cluster_id in sorted(set(labels)):
        idx = np.where(labels == cluster_id)[0]
        submatrix = matrix[np.ix_(idx, idx)]

        # Exclude diagonal self-similarity
        if len(idx) > 1:
            off_diag = submatrix[~np.eye(len(idx), dtype=bool)]
            mean_internal = float(off_diag.mean())
            max_internal = float(off_diag.max())
            min_internal = float(off_diag.min())
        else:
            mean_internal = max_internal = min_internal = 1.0

        # Medoid: element with highest average similarity to others in same cluster
        if len(idx) > 1:
            avg_sim = submatrix.mean(axis=1)
            medoid_local = int(np.argmax(avg_sim))
            medoid_global = int(idx[medoid_local])
        else:
            medoid_global = int(idx[0])

        clusters[cluster_id] = {
            "members": idx.tolist(),
            "size": len(idx),
            "mean_internal_similarity": mean_internal,
            "max_internal_similarity": max_internal,
            "min_internal_similarity": min_internal,
            "medoid_index": medoid_global,
        }

    # Rank clusters from most cohesive to least cohesive
    ranked = sorted(
        clusters.items(),
        key=lambda item: item[1]["mean_internal_similarity"],
        reverse=True,
    )
    return ranked

if __name__ == "__main__":
    matrix_path = Path("test_and_metrics/results/sentence_transformer_similarity/HPA-example_sentence_transformer_matrix.txt")
    matrix = load_matrix(matrix_path)

    labels, chosen_k = cluster_similarity_matrix(matrix, k=None, k_min=2, k_max=5)
    ranked_clusters = cluster_stats(matrix, labels)

    print("Cluster ranking:")
    for rank, (cluster_id, info) in enumerate(ranked_clusters, start=1):
        print(f"\nRank {rank} - cluster {cluster_id}")
        print(f"Size: {info['size']}")
        print(f"Members: {info['members']}")
        print(f"Mean internal similarity: {info['mean_internal_similarity']:.4f}")
        print(f"Max internal similarity: {info['max_internal_similarity']:.4f}")
        print(f"Min internal similarity: {info['min_internal_similarity']:.4f}")
        print(f"Medoid index: {info['medoid_index']}")


'''
Risultato per jaccard similarity esempio:


Silhouette score:
  k=2: 0.2578  <-- best one
  k=3: 0.2333
  k=4: 0.2143
  k=5: 0.2046

Cluster ranking:

Rank 1 - cluster 1
Size: 20
Members: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
Mean internal similarity: 0.5491
Max internal similarity: 0.8267
Min internal similarity: 0.3069
Medoid index: 19

Rank 2 - cluster 2
Size: 40
Members: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
Mean internal similarity: 0.3187
Max internal similarity: 0.4943
Min internal similarity: 0.1781
Medoid index: 30

'''