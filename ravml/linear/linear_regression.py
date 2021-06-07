import matplotlib.pyplot as plt
import ravop.core as R
from ravcom import inform_server
inform_server()

class LinearRegression():
    def __init__(self,x_points,y_points,theta):
        self.raw_X = x_points
        self.raw_y = y_points
        self.m = R.Scalar(self.raw_y.shape[0])
        self.X = R.Tensor(self.raw_X.tolist())
        self.y = R.Tensor(self.raw_y.tolist())
        self.theta = R.Tensor(theta.tolist())

    def compute_cost(self):
        residual = self.X.dot(self.theta).sub(self.y)
        while residual.status != 'computed':
            pass
        return (R.Scalar(1).div(R.Scalar(2).multiply(self.m))).multiply(residual.dot(residual.transpose()))

    def gradient_descent(self, alpha, num_iters):
        alpha_ = R.Scalar(alpha)
        for e in range(num_iters):
            residual = self.X.dot(self.theta).sub(self.y)
            while residual.status != 'computed':
                pass
            temp = self.theta.sub((alpha_.div(self.m)).multiply(self.X.transpose().dot(residual)))
            while temp.status!='computed':
                pass
            self.theta = temp
            print('Iteration : ',e)
        op_theta = self.theta()
        print('Theta found by gradient descent: intercept={0}, slope={1}'.format(op_theta[0],op_theta[1]))
        return self.theta

    def plot_graph(self,optimal_theta):
        optimal_theta = optimal_theta()
        fig, ax = plt.subplots()
        ax.plot(self.raw_X[:,1], self.raw_y[:,0], 'o', label='Raw Data')
        ax.plot(self.raw_X[:,1], self.raw_X.dot(optimal_theta), linestyle='-', label='Linear Regression')
        plt.ylabel('Profit')
        plt.xlabel('Population of City')
        legend = ax.legend(loc='upper center', shadow=True)
        plt.show()
