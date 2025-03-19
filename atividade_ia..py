import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

# =======================
# 1️⃣ Carregar os Dados
# =======================

# Base Iris
iris = load_iris()
X_iris = iris.data  # Apenas os dados, sem rótulos

# Base Netflix
df_netflix = pd.read_csv("/home/paolants/Documentos/GitHub/Agrupamento_IA/netflix.csv")

# Converter a coluna "duration" para número (minutos)
def convert_duration(value):
    if "min" in str(value):
        return int(value.split()[0])  # Pega só o número (ex: "90 min" -> 90)
    elif "Season" in str(value):
        return int(value.split()[0]) * 600  # Exemplo: 1 temporada = 600 min
    else:
        return np.nan

df_netflix["duration_numeric"] = df_netflix["duration"].apply(convert_duration)
df_netflix = df_netflix.dropna(subset=["duration_numeric"])  # Remove valores nulos

# Selecionar colunas relevantes
X_netflix = df_netflix[['release_year', 'duration_numeric']]

# =======================
# 2️⃣ Normalizar os Dados
# =======================
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)
X_netflix_scaled = scaler.fit_transform(X_netflix)

# =======================
# 3️⃣ Definir Função de Agrupamento
# =======================
def aplicar_algoritmos(X, nome_base):
    print(f"\n===== {nome_base} =====")

    # K-Means com definição do melhor K (Elbow Method)
    distortions = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    # Plot do método do cotovelo
    plt.figure()
    plt.plot(range(2, 10), distortions, marker='o')
    plt.xlabel("Número de Clusters")
    plt.ylabel("Inertia")
    plt.title(f"Elbow Method - {nome_base}")
    plt.show()

    # Escolher K ótimo (exemplo: 3)
    k_optimal = 3
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    clusters_kmeans = kmeans.fit_predict(X)
    
    # Bisecting K-Means
    bisect_kmeans = BisectingKMeans(n_clusters=k_optimal, random_state=42)
    clusters_bisecting = bisect_kmeans.fit_predict(X)

    # Clustering Hierárquico
    hierarquico = AgglomerativeClustering(n_clusters=k_optimal)
    clusters_hierarquico = hierarquico.fit_predict(X)

    # Mostrar Silhouette Score
    print(f"K-Means Silhouette: {silhouette_score(X, clusters_kmeans):.2f}")
    print(f"Bisecting K-Means Silhouette: {silhouette_score(X, clusters_bisecting):.2f}")
    print(f"Hierarchical Clustering Silhouette: {silhouette_score(X, clusters_hierarquico):.2f}")

    # Plot dos clusters K-Means
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=clusters_kmeans, cmap="viridis", alpha=0.6)
    plt.title(f"K-Means Clusters - {nome_base}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Plot do dendrograma para agrupamento hierárquico
    plt.figure()
    dendrogram(linkage(X, method='ward'))
    plt.title(f"Dendrograma - {nome_base}")
    plt.show()

# =======================
# 4️⃣ Aplicar os Algoritmos
# =======================
aplicar_algoritmos(X_iris_scaled, "Iris")
aplicar_algoritmos(X_netflix_scaled, "Netflix")
