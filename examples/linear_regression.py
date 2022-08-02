from dotenv import load_dotenv

load_dotenv()

import numpy as np
import ravop as R

from ravml.linear.linear_regression import LinearRegression


def preprocess(data):
    x = data[:, 0]
    y = data[:, 1]
    y = y.reshape(y.shape[0], 1)
    x = np.c_[np.ones(x.shape[0]), x]
    theta = np.zeros((2, 1))
    return x, y, theta


R.initialize("<Ravauth Token>")
R.flush()
R.Graph(name='linreg', algorithm='linrig', approach='distributed')
iterations = 5
alpha = 0.01
data = np.loadtxt('data/data_linreg.txt', delimiter=',')

x, y, theta = preprocess(data)

model = LinearRegression(x, y, theta)
model.compute_cost()
theta = model.gradient_descent(alpha, iterations)
model.predict(x)
print(y)
model.score(x, y)

R.activate()

R.execute()
R.track_progress()
optimal_theta = R.fetch_persisting_op(op_name="theta")
pred = R.fetch_persisting_op(op_name="predicted values")
print("optimized theta", optimal_theta)
print("predicted values:", pred)
# print(y)
model.plot_graph(optimal_theta=optimal_theta)
