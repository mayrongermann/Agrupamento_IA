import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.preprocessing import StandardScaler

iris = pd.read_csv('iris.csv')
netflix = pd.read_csv('netflix.csv')

x1 = iris[['sepal.length', 'sepal.width']]
x2 = netflix[['show_id','duration']]

scaler = StandardScaler()

x1_scaled = scaler.fit_transform(x1)
x2_scaled = scaler.fit_transform(x2)

num_clusters = 3


kmeans1 = KMeans(n_clusters=num_clusters, random_state=42)
iris['kmeans_cluster'] = kmeans1.fit_predict(X1_scaled)

kmeans2 = KMeans(n_clusters=num_clusters, random_state=42)
netflix['kmeans_cluster'] = kmeans2.fit_predict(X2_scaled)


bisecting_kmeans1 = BisectingKMeans(n_clusters=num_clusters, random_state=42)
iris['bisecting_cluster'] = bisecting_kmeans1.fit_predict(X1_scaled)

bisecting_kmeans2 = BisectingKMeans(n_clusters=num_clusters, random_state=42)
netflix['bisecting_cluster'] = bisecting_kmeans2.fit_predict(X2_scaled)

def plot_clusters(X, labels, title, fig_num):
    plt.figure(fig_num)  # Criar uma aba separada
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid()

plot_clusters(X1_scaled, iris['kmeans_cluster'], "K-Means - Dataset 1", 1)
plot_clusters(X1_scaled, iris['bisecting_cluster'], "Bisecting K-Means - Dataset 1", 2)
plot_clusters(X2_scaled, netflix['kmeans_cluster'], "K-Means - Dataset 2", 3)
plot_clusters(X2_scaled, netflix['bisecting_cluster'], "Bisecting K-Means - Dataset 2", 4)

plt.show()