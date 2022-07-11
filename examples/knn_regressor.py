from ravml.neighbors.knn import KNNRegressor
import ravop.core as R

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split






algo = R.Graph(name='knn', algorithm='knn', approach='distributed')

knn = KNNRegressor()
iris = load_iris()

X = iris.data[:700]
y = iris.target[:700]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
import numpy as np
data = np.loadtxt('data/data_linreg.txt', delimiter=',')

knn.fit(X_train,y_train, n_neighbours=5)
print(knn.predict(X_test)())
print(knn.score(y_test=y_test)())
# inform_server()

algo.end()