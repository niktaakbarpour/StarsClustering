import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


def main():
    X, Y = scan('../raw/Stars.csv')
    X = preprocessing(X)

    draw_wcss(X)
    draw_dendrogram(X)

    k_means(X, Y)
    agglomerative(X, Y)


def scan(file_name):
    dataset = pd.read_csv(file_name)
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    return X, Y


def preprocessing(X):
    ct = ColumnTransformer(transformers=[
        ("encoder", OneHotEncoder(sparse=False), [4, 5]),
        ("discretizer", KBinsDiscretizer(n_bins=[3, 3, 3, 3], encode='onehot-dense', strategy='quantile'), [0, 1, 2, 3])
    ], remainder="passthrough")
    X = np.array(ct.fit_transform(X))

    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X


def k_means(X, Y):
    cluster = KMeans(n_clusters=6, init='k-means++', random_state=42)
    Y_pred = cluster.fit_predict(X)
    Y_pred = adapt(Y_pred, Y)
    result(accuracy_score(Y, Y_pred), 'K-Means')


def agglomerative(X, Y):
    cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
    Y_pred = cluster.fit_predict(X)
    Y_pred = adapt(Y_pred, Y)
    result(accuracy_score(Y, Y_pred), 'Hierarchical')


def result(accuracy, algorithm):
    print(f"{algorithm}:")
    print(f"Accuracy= {accuracy}")
    print("--------------------------")


def adapt(Y_pred, Y):
    newY_pred = np.array(range(len(Y_pred)))
    for i in range(0, 6):
        data = Counter(Y[Y_pred == i])
        newY_pred[Y_pred == i] = data.most_common(1)[0][0]
    return newY_pred


def draw_wcss(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def draw_dendrogram(X):
    sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Clusters')
    plt.ylabel('Euclidean distances')
    plt.show()


if __name__ == '__main__':
    main()
