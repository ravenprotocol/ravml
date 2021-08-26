from ravml.linear.linear_regression import LinearRegression
import numpy as np

def preprocess(data):
    x = data[:,0]
    y = data[:,1]
    y = y.reshape(y.shape[0], 1)
    x = np.c_[np.ones(x.shape[0]), x] # adding column of ones to X to account for theta_0 (the intercept)
    theta = np.zeros((2, 1))
    return x,y,theta

iterations = 20
alpha = 0.01

data = np.loadtxt('examples/data_linreg.txt', delimiter=',')
x,y,theta = preprocess(data)

model = LinearRegression(x,y,theta)
model.compute_cost()            # initial cost with coefficients at zero
optimal_theta = model.gradient_descent(alpha, iterations)
model.plot_graph(optimal_theta)

