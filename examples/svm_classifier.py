import numpy as np

X, y =  datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = list(np.where(y == 0, -1, 1))

clf = SVM()
clf.fit(X, y)