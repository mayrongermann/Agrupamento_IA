import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine
from kneed import KneeLocator
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import matplotlib

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
X_iris_scaled = scaler.fit_transform(df_iris)
X_wine_scaled = scaler.fit_transform(df_wine)

pca = PCA(n_components=3)
X_iris_pca = pca.fit_transform(X_iris_scaled)
X_wine_pca = pca.fit_transform(X_wine_scaled)

def plot_elbow_curve(X, dataset_name):
    distortions = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    kn = KneeLocator(K_range, distortions, curve="convex", direction="decreasing")
    
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, distortions, marker='o', linestyle='--', color='b')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Distorção (Inércia)')
    plt.title(f"Gráfico de Cotovelo - {dataset_name}")
    plt.axvline(x=kn.elbow, color='r', linestyle='--', label=f"Cotovelo: k={kn.elbow}")
    plt.legend()
    plt.grid()
    plt.savefig(f"elbow_curve_{dataset_name.lower()}.png")
    plt.show()
    return kn.elbow

def plot_correlation_matrix(df, dataset_name):
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Matriz de Correlação - {dataset_name}")
    plt.savefig(f"correlation_matrix_{dataset_name.lower()}.png")
    plt.show()

def plot_pairplot(df, dataset_name):
    sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle(f"Pairplot - {dataset_name}", y=1.02)
    plt.savefig(f"pairplot_{dataset_name.lower()}.png")
    plt.show()

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
    axes[0].set_xlabel("Componente 1")
    axes[0].set_ylabel("Componente 2")

    axes[1].scatter(X[:, 0], X[:, 1], c=clusters_bisecting, cmap="viridis", alpha=0.6)
    axes[1].set_title(f"Bisecting K-Means Clusters - {dataset_name}")
    axes[1].set_xlabel("Componente 1")
    axes[1].set_ylabel("Componente 2")
    
    Z = linkage(X, method='ward')
    dendrogram(Z, ax=axes[2])
    axes[2].set_title(f"Dendrograma - {dataset_name}")
    
    plt.tight_layout()
    plt.savefig(f"clustering_{dataset_name.lower()}.png")
    plt.show()

def plot_silhouette(X, labels, dataset_name):
    silhouette_vals = silhouette_samples(X, labels)
    cluster_labels = np.unique(labels)
    n_clusters = cluster_labels.shape[0]
    y_lower, y_upper = 0, 0
    yticks = []

    plt.figure(figsize=(10, 6))
    for i, cluster in enumerate(cluster_labels):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1.0)
        yticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)

    avg_score = np.mean(silhouette_vals)
    plt.axvline(avg_score, color="red", linestyle="--", label=f"Média: {avg_score:.2f}")
    plt.yticks(yticks, [f"Cluster {i+1}" for i in range(n_clusters)])
    plt.ylabel("Clusters")
    plt.xlabel("Coeficiente de Silhueta")
    plt.title(f"Gráfico de Silhueta - {dataset_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"silhouette_{dataset_name.lower()}.png")
    plt.show()

print("\n=== Gerando gráficos para o dataset Iris ===")
k_iris = plot_elbow_curve(X_iris_scaled, "Iris")
plot_correlation_matrix(df_iris, "Iris")
plot_pairplot(df_iris, "Iris")
kmeans_iris = KMeans(n_clusters=k_iris, random_state=42, n_init=10)
clusters_kmeans_iris = kmeans_iris.fit_predict(X_iris_pca)
plot_silhouette(X_iris_pca, clusters_kmeans_iris, "Iris")

bisect_kmeans_iris = BisectingKMeans(n_clusters=k_iris, random_state=42, n_init=10)
clusters_bisecting_iris = bisect_kmeans_iris.fit_predict(X_iris_pca)
plot_silhouette(X_iris_pca, clusters_bisecting_iris, "Iris (Bisecting K-Means)")
apply_clustering(X_iris_pca, k_iris, "Iris", feature_names_iris)

print("\n=== Gerando gráficos para o dataset Wine ===")
k_wine = plot_elbow_curve(X_wine_scaled, "Wine")
plot_correlation_matrix(df_wine, "Wine")
plot_pairplot(df_wine, "Wine")
kmeans_wine = KMeans(n_clusters=k_wine, random_state=42, n_init=10)
clusters_kmeans_wine = kmeans_wine.fit_predict(X_wine_pca)
plot_silhouette(X_wine_pca, clusters_kmeans_wine, "Wine")

bisect_kmeans_wine = BisectingKMeans(n_clusters=k_wine, random_state=42, n_init=10)
clusters_bisecting_wine = bisect_kmeans_wine.fit_predict(X_wine_pca)
plot_silhouette(X_wine_pca, clusters_bisecting_wine, "Wine (Bisecting K-Means)")
apply_clustering(X_wine_pca, k_wine, "Wine", feature_names_wine)