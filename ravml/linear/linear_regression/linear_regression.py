
import ravop.core  as R
import numpy as np
import pathlib
from ravml.metrics import metrics
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self,x_points,y_points,theta):
        self.raw_X = x_points
        self.raw_y = y_points
        self.m = R.t(self.raw_y.shape[0])
        self.X = R.t(self.raw_X.tolist())
        self.y = R.t(self.raw_y.tolist())
        self.theta = R.t(theta.tolist())

    def compute_cost(self):
        residual = self.X.dot(self.theta).sub(self.y)
        return (R.t(1).div(R.t(2).multiply(self.m))).multiply(residual.dot(residual.transpose()))

    def gradient_descent(self, alpha, num_iters):
        alpha_ = R.t(alpha)
        for e in range(1,num_iters+1):
            residual = self.X.dot(self.theta).sub(self.y)
            temp = self.theta.sub((alpha_.div(self.m)).multiply(self.X.transpose().dot(residual)))
            self.theta = R.t(temp())
            print('Iteration : ',e)
        op_theta = self.theta()
        print('Theta found by gradient descent: intercept={0}, slope={1}'.format(op_theta[0],op_theta[1]))
        return self.theta, op_theta[0], op_theta[1]

    def plot_graph(self,optimal_theta, res_file_path):
        optimal_theta = optimal_theta()
        fig, ax = plt.subplots()
        ax.plot(self.raw_X[:,1], self.raw_y[:,0], 'o', label='Raw Data')
        ax.plot(self.raw_X[:,1], self.raw_X.dot(optimal_theta), linestyle='-', label='Linear Regression')

        plt.savefig(res_file_path)
        plt.show()

    def score(self, X, y, name="r2"):
        if not isinstance(X, R.Tensor):
            X = R.Tensor(X)
        if not isinstance(y, R.Tensor):
            y = R.Tensor(y)

        y_pred = self.predict(X)
        y_true = y

        if name == "r2":
            return metrics.r2_score(y_true, y_pred)
        else:
            return None

    def set_params(self, **kwargs):
        param_dict={
            'theta': self.theta(),
            'X':self.X(),
            'y':self.y()
        }
        for i in kwargs.keys():
            if i in param_dict.keys():
                param_dict[i]=kwargs[i]

        return param_dict

        
    def get_params(self):
        param_dict={
            'theta': self.theta(),
            'X':self.X(),
            'y':self.y()
        }
        return param_dict