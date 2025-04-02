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
import urllib.request

# Baixar o dataset Seeds diretamente da URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
urllib.request.urlretrieve(url, "seeds_dataset.txt")

# Carregar o dataset Seeds
data = pd.read_csv("seeds_dataset.txt", sep='\s+', header=None)

# Nomeando as colunas
columns = ['Area', 'Perimeter', 'Compactness', 'Kernel Length', 'Kernel Width',
           'Asymmetry Coefficient', 'Kernel Groove Length', 'Class']
data.columns = columns

# Separar features (apenas as variáveis explicativas)
X_seeds = data.iloc[:, :-1]

# Calcular a matriz de correlação
corr_matrix = X_seeds.corr()

# Plotar a matriz de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação do Dataset Seeds")
plt.show()

# Exibir a matriz de correlação no terminal
print("Matriz de Correlação do Dataset Seeds:")
print(corr_matrix)

# Normalizar os dados do dataset Seeds
scaler = StandardScaler()
X_seeds_normalized = scaler.fit_transform(X_seeds)

# Gerar o pairplot do dataset Seeds
def generate_pairplot_from_dataframe(df, dataset_name):
    # Normalizar os dados
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Gerar o pairplot
    sns.pairplot(df_normalized, diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle(f"Pairplot do Dataset {dataset_name} (Normalizado)", y=1.02)
    plt.show()

generate_pairplot_from_dataframe(X_seeds, "Seeds")

# Carregar os dados Iris e Wine
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

# Exibir informações básicas
print("Formato da base Iris:", df_iris.shape)
print("Formato da base Wine:", df_wine.shape)

print("Cabeçalho da base Iris:")
print(df_iris.head())
print("Cabeçalho da base Wine:")
print(df_wine.head())

# Tratar dados faltantes
print("Dados faltantes na base Iris:")
print(df_iris.isnull().sum())
print("Dados faltantes na base Wine:")
print(df_wine.isnull().sum())
df_iris.fillna(0, inplace=True)
df_wine.fillna(0, inplace=True)

# Padronizar os dados
scaler = StandardScaler()
X_iris = scaler.fit_transform(df_iris)
X_wine = scaler.fit_transform(df_wine)

# Função para plotar os dados originais
def plot_raw_data(df, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.6)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title(f"Distribuição Original dos Dados - {dataset_name}")
    plt.show()

plot_raw_data(df_iris, "Iris")
plot_raw_data(df_wine, "Wine")

# Reduzir dimensionalidade com PCA
pca = PCA(n_components=3)
X_iris_pca = pca.fit_transform(X_iris)
X_wine_pca = pca.fit_transform(X_wine)

# Função para encontrar o número ideal de clusters
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

# Função para aplicar os algoritmos de clustering
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
    plt.show()

    
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
    plt.show()
    
    return kn.elbow

# Função para gerar o gráfico de silhueta
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
    plt.show()

# Função para gerar e exibir a matriz de confusão
def plot_confusion_matrix(y_true, y_pred, dataset_name):
    # Ajustar os rótulos previstos para corresponderem aos rótulos verdadeiros
    contingency_matrix = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    adjusted_pred = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        adjusted_pred[y_pred == j] = i

    # Gerar a matriz de confusão ajustada
    adjusted_confusion_matrix = confusion_matrix(y_true, adjusted_pred)

    # Exibir a matriz de confusão no formato desejado
    print(f"\nMatriz de Confusão - {dataset_name}:")
    print(f"{adjusted_confusion_matrix}")

    # Plotar a matriz de confusão
    plt.figure(figsize=(6, 5))
    plt.imshow(adjusted_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {dataset_name}")
    plt.colorbar()
    plt.xticks(np.arange(len(np.unique(y_true))), np.unique(y_true))
    plt.yticks(np.arange(len(np.unique(y_true))), np.unique(y_true))
    plt.ylabel('Rótulos Verdadeiros')
    plt.xlabel('Rótulos Previstos')

    # Adicionar os números em cada bloco
    for i in range(adjusted_confusion_matrix.shape[0]):
        for j in range(adjusted_confusion_matrix.shape[1]):
            plt.text(j, i, adjusted_confusion_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if adjusted_confusion_matrix[i, j] > adjusted_confusion_matrix.max() / 2 else "black")

    plt.tight_layout()
    plt.show()

k_iris = plot_elbow_curve(X_iris_pca, "Iris")
k_wine = plot_elbow_curve(X_wine_pca, "Wine")

print(f"Número ideal de clusters para Iris: {k_iris}")
print(f"Número ideal de clusters para Wine: {k_wine}")

# Aplicar clustering nos dados Iris
kmeans_iris = KMeans(n_clusters=k_iris, random_state=42, n_init=10).fit(X_iris_pca)
apply_clustering(X_iris_pca, k_iris, "Iris", feature_names_iris)
plot_silhouette(X_iris_pca, kmeans_iris.labels_, "Iris")
plot_confusion_matrix(y_iris, kmeans_iris.labels_, "Iris")

# Aplicar clustering nos dados Wine
kmeans_wine = KMeans(n_clusters=k_wine, random_state=42, n_init=10).fit(X_wine_pca)
apply_clustering(X_wine_pca, k_wine, "Wine", feature_names_wine)
plot_silhouette(X_wine_pca, kmeans_wine.labels_, "Wine")
plot_confusion_matrix(y_wine, kmeans_wine.labels_, "Wine")
