import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine
from kneed import KneeLocator
from sklearn.decomposition import PCA

data_iris = load_iris()
X_iris = data_iris['data']
y_iris = data_iris['target']
feature_names_iris = data_iris['feature_names']

data_wine = load_wine()
X_wine = data_wine['data']
y_wine = data_wine['target']
feature_names_wine = data_wine['feature_names']

df_iris = pd.DataFrame(X_iris, columns=feature_names_iris)
df_wine = pd.DataFrame(X_wine, columns=feature_names_wine)

print("Formato da base Iris:", df_iris.shape)
print("Formato da base Wine:", df_wine.shape)

print("Cabeçalho da base Iris:")
print(df_iris.head())
print("Cabeçalho da base Wine:")
print(df_wine.head())

print("Dados faltantes na base Iris:")
print(df_iris.isnull().sum())
print("Dados faltantes na base Wine:")
print(df_wine.isnull().sum())
df_iris.fillna(0, inplace=True)
df_wine.fillna(0, inplace=True)

scaler = StandardScaler()
X_iris = scaler.fit_transform(df_iris)
X_wine = scaler.fit_transform(df_wine)

def plot_raw_data(df, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.6)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title(f"Distribuição Original dos Dados - {dataset_name}")
    plt.show()

plot_raw_data(df_iris, "Iris")
plot_raw_data(df_wine, "Wine")

pca = PCA(n_components=3)
X_iris_pca = pca.fit_transform(X_iris)
X_wine_pca = pca.fit_transform(X_wine)

def find_optimal_k(X):
    distortions = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    kn = KneeLocator(K_range, distortions, curve="convex", direction="decreasing")
    return kn.elbow

k_iris = find_optimal_k(X_iris_pca)
k_wine = find_optimal_k(X_wine_pca)
print(f"Número ideal de clusters para Iris: {k_iris}")
print(f"Número ideal de clusters para Wine: {k_wine}")

def apply_clustering(X, k, dataset_name, feature_names):
    print(f"\n===== {dataset_name} =====")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_kmeans = kmeans.fit_predict(X)

    bisect_kmeans = BisectingKMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_bisecting = bisect_kmeans.fit_predict(X)

    print(f"K-Means Silhouette Score: {silhouette_score(X, clusters_kmeans):.2f}")
    print(f"Bisecting K-Means Silhouette Score: {silhouette_score(X, clusters_bisecting):.2f}")
    print(f"K-Means Davies-Bouldin Score: {davies_bouldin_score(X, clusters_kmeans):.2f}")
    print(f"K-Means Calinski-Harabasz Score: {calinski_harabasz_score(X, clusters_kmeans):.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=clusters_kmeans, cmap="viridis", alpha=0.6)
    axes[0].set_title(f"K-Means Clusters - {dataset_name}")
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])

    axes[1].scatter(X[:, 0], X[:, 1], c=clusters_bisecting, cmap="viridis", alpha=0.6)
    axes[1].set_title(f"Bisecting K-Means Clusters - {dataset_name}")
    axes[1].set_xlabel(feature_names[0])
    axes[1].set_ylabel(feature_names[1])
    
    Z = linkage(X, method='ward')
    dendrogram(Z, ax=axes[2])
    axes[2].set_title(f"Dendrograma - {dataset_name}")
    
    plt.tight_layout()
    plt.show()

apply_clustering(X_iris_pca, k_iris, "Iris", feature_names_iris)
apply_clustering(X_wine_pca, k_wine, "Wine", feature_names_wine)
