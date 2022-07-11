from ravml.linear.logistic_regression import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :2]
y = (iris.target != 0) * 1

model = LogisticRegression(lr=0.1, num_iter=30)

model.fit(X, y)

preds = model.predict(X)
print((preds == y).mean())
print(model.theta())

model.plot_loss()

model.visualize(X,y)
