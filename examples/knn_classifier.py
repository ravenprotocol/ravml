from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()
import ravop as R
from sklearn.datasets import load_iris
from ravml.neighbors import KNNClassifier

R.initialize("<token>")
R.flush()
R.Graph(name='knn', algorithm='knn', approach='distributed')

iris = load_iris()
X = iris.data[:5500]
y = iris.target[:5500]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
knn = KNNClassifier()
knn.fit(X_train, y_train, n_neighbours=20, n_classes=3)
knn.predict(X_test)
knn.score(y_test=y_test)

R.activate()

R.execute()
R.track_progress()
accuracy = R.fetch_persisting_op(op_name="accuracy")
label = R.fetch_persisting_op(op_name="label")
print("accuracy : ", accuracy)
print("predicted label : ", label)
