from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ravml.cluster import MiniBatchKMeans
import ravop.core as R

algo = R.Graph(name='kmeans', algorithm='kmeans', approach='distributed')


k = MiniBatchKMeans()

iris = load_iris()

X = iris.data[:100]
y = iris.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
k.fit(X_train, 3, iter=30)
k.plot()

algo.end()