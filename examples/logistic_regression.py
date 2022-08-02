from dotenv import load_dotenv

load_dotenv()

from ravml.linear.logistic_regression import LogisticRegression
import ravop as R
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X = iris.data[:3500]
y = iris.target[:3500]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)

R.initialize("<token>")
R.flush()
R.Graph(name='logrig', algorithm='logrig', approach='distributed')

model = LogisticRegression(lr=0.1, num_iter=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)

R.activate()

R.execute()
R.track_progress()
label = R.fetch_persisting_op(op_name="predicted_vals")
print("predicted label : ", label)
