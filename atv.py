import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.preprocessing import StandardScaler

# Carregar os dados
data_x = load_iris()
df_iris = data_x['data']  # Acesso correto ao atributo 'data'
data_y = load_wine()
df_wine = data_y['data']  # Acesso correto ao atributo 'data'

# Selecionar as duas primeiras colunas para visualização
X_iris = df_iris[:, :2]  # Usando numpy para acessar as colunas diretamente
X_wine = df_wine[:, :2]  # Usando numpy para acessar as colunas diretamente

# Padronizar os dados para melhorar o clustering
scaler = StandardScaler()
X_iris = scaler.fit_transform(X_iris)
X_wine = scaler.fit_transform(X_wine)

# Aplicar KMeans e Bisecting KMeans nos dois conjuntos de dados
kmeans_iris = KMeans(n_clusters=5, random_state=42)
kmeans_iris.fit(X_iris)
y_kmeans_iris = kmeans_iris.predict(X_iris)

bisect_kmeans_iris = BisectingKMeans(n_clusters=5, random_state=42)
bisect_kmeans_iris.fit(X_iris)
y_bisect_kmeans_iris = bisect_kmeans_iris.predict(X_iris)

kmeans_wine = KMeans(n_clusters=5, random_state=42)
kmeans_wine.fit(X_wine)
y_kmeans_wine = kmeans_wine.predict(X_wine)

bisect_kmeans_wine = BisectingKMeans(n_clusters=5, random_state=42)
bisect_kmeans_wine.fit(X_wine)
y_bisect_kmeans_wine = bisect_kmeans_wine.predict(X_wine)

# Criar subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plotar KMeans para Iris
axes[0, 0].scatter(X_iris[:, 0], X_iris[:, 1], c=y_kmeans_iris, cmap='viridis')
axes[0, 0].set_title('KMeans - Iris')
axes[0, 0].set_xlabel("Altura Sépala (padronizada)")
axes[0, 0].set_ylabel("Largura Sépala (padronizada)")

# Plotar Bisecting KMeans para Iris
axes[0, 1].scatter(X_iris[:, 0], X_iris[:, 1], c=y_bisect_kmeans_iris, cmap='viridis')
axes[0, 1].set_title('Bisecting KMeans - Iris')
axes[0, 1].set_xlabel("Altura Sépala")
axes[0, 1].set_ylabel("Largura Sépala")

# Plotar KMeans para Wine
axes[1, 0].scatter(X_wine[:, 0], X_wine[:, 1], c=y_kmeans_wine, cmap='viridis')
axes[1, 0].set_title('KMeans - Wine')
axes[1, 0].set_xlabel("Alcohol")
axes[1, 0].set_ylabel("Malic acid")

# Plotar Bisecting KMeans para Wine
axes[1, 1].scatter(X_wine[:, 0], X_wine[:, 1], c=y_bisect_kmeans_wine, cmap='viridis')
axes[1, 1].set_title('Bisecting KMeans - Wine')
axes[1, 1].set_xlabel("Alcohol")
axes[1, 1].set_ylabel("Malic acid")

plt.tight_layout()
plt.show()
