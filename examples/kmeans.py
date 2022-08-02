import os

from dotenv import load_dotenv

load_dotenv()

from ravml.cluster import KMeans

import ravop as R

R.initialize(os.environ.get('TOKEN'))
R.flush()
R.Graph(name='kmeans', algorithm='kmeans', approach='distributed')

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:1000]
y = iris.target[:1000]
print(y)
k = KMeans()
k.fit(X, 3, iter=30)
R.activate()

R.execute()
R.track_progress()
label = R.fetch_persisting_op(op_name="kmeans_label")
print("predicted label : ", label)
