from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ravml.cluster import KMeans

if __name__ == '__main__':
    k = KMeans()

    iris = load_iris()

    X = iris.data[:1000]
    y = iris.target[:1000]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
    k.fit(X_train, 3, iter=30)
    k.plot()
