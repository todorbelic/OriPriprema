from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from kmeans import KMeans
import pandas as pd
import numpy as np

# --- UCITAVANJE I PRIKAZ IRIS DATA SETA --- #

iris_data = load_iris()  # ucitavanje Iris data seta
iris_data = iris_data.data[:, 1:3]  # uzima se druga i treca osobina iz data seta (sirina sepala i duzina petala)
dataf = pd.read_csv('../../customer_churn.csv')
print(np.count_nonzero(dataf['churn']))

data = dataf[['total intl minutes', 'total day minutes']]
data = data.values
dataf = dataf.values

plt.figure()
for i in range(len(data)):
    plt.scatter(data[i, 0], data[i, 1])

plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.show()


# --- INICIJALIZACIJA I PRIMENA K-MEANS ALGORITMA --- #

# TODO 2: K-means na Iris data setu
kmeans = KMeans(n_clusters=2, max_iter=100)
kmeans.fit(data, dataf, normalize=True)

colors = {0: 'red', 1: 'green', 2: 'black', 3: 'orange'}
plt.figure()
for idx, cluster in enumerate(kmeans.clusters):
    plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
    for datum in cluster.data:  # iscrtavanje tacaka
        plt.scatter(datum[0], datum[1], c=colors[idx])

for cluster in kmeans.clusters:
    print(cluster.get_churn())

plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.show()


# --- ODREDJIVANJE OPTIMALNOG K --- #

plt.figure()
sum_squared_errors = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100)
    kmeans.fit(data, dataf)
    sse = kmeans.sum_squared_error()
    sum_squared_errors.append(sse)

plt.plot([x for x in range(2, 10)], sum_squared_errors)
plt.xlabel('# of clusters')
plt.ylabel('SSE')
plt.show()


# TODO 7: DBSCAN nad Iris podacima, prikazati rezultate na grafiku isto kao kod K-means
