import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

def evaluate_clustering_metrics(embeddings, labels, num_clusters):
   
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    predicted_clusters = kmeans.fit_predict(embeddings)
    ari = metrics.adjusted_rand_score(labels, predicted_clusters)
    ami = metrics.adjusted_mutual_info_score(labels, predicted_clusters)

    return ari, ami

def visualize_embeddings(embeddings, labels, visualize_output_dir):
    tsne = TSNE(n_components=2, random_state=42)
    embedded_embeddings = tsne.fit_transform(embeddings)

    num_unique_labels = len(np.unique(labels))

    colors = plt.cm.get_cmap('tab20', num_unique_labels)

    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        class_embeddings = embedded_embeddings[class_indices]
        plt.scatter(
            class_embeddings[:, 0],
            class_embeddings[:, 1],
            color=colors(class_label),
            label=f'Class {class_label}'
        )
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Visualization of Embeddings using t-SNE')
    plt.legend()

    plt.savefig(f'{visualize_output_dir}/embeddings_visualization.png')
    plt.close()
    print(f"Saved TSNE plot to {visualize_output_dir}\n")

def visualize_pca_kmeans(embeddings, num_classes,visualize_output_dir):
    
   
 
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    num_clusters = num_classes
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(pca_result)
    cluster_centers = kmeans.cluster_centers_

    plt.figure(figsize=(10, 6))

    for cluster_label in range(num_clusters):
        plt.scatter(
            pca_result[cluster_labels == cluster_label, 0],
            pca_result[cluster_labels == cluster_label, 1],
            label=f'Cluster {cluster_label}',
            alpha=0.7
        )

    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c='black',
        marker='x',
        s=100,
        label='Centroids'
    )

    plt.title('PCA + KMeans Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig(f'{visualize_output_dir}/embeddings_pca_kmeans.png')
    plt.close()
    print(f" Saved PCA+KMeans plot to {visualize_output_dir}\n")
