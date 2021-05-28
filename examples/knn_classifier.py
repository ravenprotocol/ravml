from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ravml.classifier import KNNClassifier

# from ravcom import inform_server


if __name__ == '__main__':
    knn = KNNClassifier()
    iris = load_iris()

    X = iris.data[:700]
    y = iris.target[:700]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)

    knn.fit(X_train, y_train, n_neighbours=5, n_classes=3)

    print(knn.predict(X_test))

    print(knn.score(y_test=y_test))
    # inform_server()
