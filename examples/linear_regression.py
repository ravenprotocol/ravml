from ravml.linear.linear_regression.linear_regression import LinearRegression
import numpy as np
import pathlib


def preprocess(data):
    x = data[:,0]
    y = data[:,1]
    y = y.reshape(y.shape[0], 1)
    x = np.c_[np.ones(x.shape[0]), x] # adding column of ones to X to account for theta_0 (the intercept)
    theta = np.zeros((2, 1))
    return x,y,theta

iterations = 20
alpha = 0.01

data = np.loadtxt('data_linreg.txt', delimiter=',')

x,y,theta = preprocess(data)

model = LinearRegression(x,y,theta)
model.compute_cost()            # initial cost with coefficients at zero
optimal_theta, inter, slope = model.gradient_descent(alpha, iterations)
print(optimal_theta, inter, slope)
res_file_path = str(pathlib.Path().resolve()) + '/result.png'
print(res_file_path)
model.plot_graph(optimal_theta, res_file_path)

