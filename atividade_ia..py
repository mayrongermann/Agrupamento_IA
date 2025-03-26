import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine
from kneed import KneeLocator

# ============================
# 1️⃣ Carregar os Dados
# ============================
data_iris = load_iris()
X_iris = data_iris['data']
y_iris = data_iris['target']
feature_names_iris = data_iris['feature_names']

data_wine = load_wine()
X_wine = data_wine['data']
y_wine = data_wine['target']
feature_names_wine = data_wine['feature_names']

# ============================
# 2️⃣ Normalizar os Dados
# ============================
scaler = StandardScaler()
X_iris = scaler.fit_transform(X_iris)
X_wine = scaler.fit_transform(X_wine)

# ============================
# 3️⃣ Definir Melhor Número de Clusters (Elbow Method)
# ============================
def find_optimal_k(X):
    distortions = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    kn = KneeLocator(K_range, distortions, curve="convex", direction="decreasing")
    return kn.elbow

k_iris = find_optimal_k(X_iris)
k_wine = find_optimal_k(X_wine)
print(f"Número ideal de clusters para Iris: {k_iris}")
print(f"Número ideal de clusters para Wine: {k_wine}")

# ============================
# 4️⃣ Aplicar Algoritmos de Agrupamento
# ============================
def apply_clustering(X, k, dataset_name, feature_names):
    print(f"\n===== {dataset_name} =====")
    
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_kmeans = kmeans.fit_predict(X)

    # Bisecting K-Means
    bisect_kmeans = BisectingKMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_bisecting = bisect_kmeans.fit_predict(X)

    # Avaliação das Métricas
    print(f"K-Means Silhouette Score: {silhouette_score(X, clusters_kmeans):.2f}")
    print(f"Bisecting K-Means Silhouette Score: {silhouette_score(X, clusters_bisecting):.2f}")
    print(f"K-Means Davies-Bouldin Score: {davies_bouldin_score(X, clusters_kmeans):.2f}")
    print(f"K-Means Calinski-Harabasz Score: {calinski_harabasz_score(X, clusters_kmeans):.2f}")

    # Criar Gráficos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot K-Means
    axes[0].scatter(X[:, 0], X[:, 1], c=clusters_kmeans, cmap="viridis", alpha=0.6)
    axes[0].set_title(f"K-Means Clusters - {dataset_name}")
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])

    # Plot Bisecting K-Means
    axes[1].scatter(X[:, 0], X[:, 1], c=clusters_bisecting, cmap="viridis", alpha=0.6)
    axes[1].set_title(f"Bisecting K-Means Clusters - {dataset_name}")
    axes[1].set_xlabel(feature_names[0])
    axes[1].set_ylabel(feature_names[1])
    
    from scipy.cluster.hierarchy import linkage
    Z = linkage(X, method='ward')
    dendrogram(Z, ax=axes[2])
    axes[2].set_title(f"Dendrograma - {dataset_name}")
    
    plt.tight_layout()
    plt.show()

# Aplicar nos datasets
apply_clustering(X_iris, k_iris, "Iris", feature_names_iris)
apply_clustering(X_wine, k_wine, "Wine", feature_names_wine)
