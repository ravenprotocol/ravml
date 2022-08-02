from dotenv import load_dotenv

load_dotenv()

from ravml.neighbors import KNNRegressor
import ravop as R

R.initialize("<TOKEN>")
R.flush()
R.Graph(name='knn', algorithm='knn', approach='distributed')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data[:1500]
y = iris.target[:1500]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)

knn = KNNRegressor()
knn.fit(X_train, y_train, n_neighbours=5)
knn.predict(X_test)
knn.score(y_test=y_test)

R.activate()

R.execute()
R.track_progress()
score = R.fetch_persisting_op(op_name="r2score_knn_classifier")
label = R.fetch_persisting_op(op_name="predicted_label")
print("R2_score : ", score)
print("predicted label : ", label)
