from ravml.classifier import KNNClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from ravcom import inform_server


if __name__ == '__main__':
    knn = KNNClassifier()
    print(knn.id)

    iris = load_digits()

    X = iris.data[:200]
    y = iris.target[:200]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)

    knn.fit(X_train, y_train, n_neighbours=5, n_classes=3)

    print(knn.predict(X_test))

    print(knn.score(y_test=y_test).id)

    inform_server()
