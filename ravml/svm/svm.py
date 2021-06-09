import ravop.core as R
from ravop.core import Scalar, Tensor, Graph
import numpy as np

class SVM(Graph):
    def __init__(self, learning_rate = 0.001, lambda_param = 0.01, n_iters = 5):

        self.lr = Scalar(learning_rate)
        self.lambda_param = Scalar(lambda_param)
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        y_ = y
        # y_ = R.where(y <= 0, -1, 1)
        self.w = R.Tensor(np.zeros(n_features))
        self.b = Scalar(0)

        for epoch in range(self.n_iters):
            print("Epoch: ",  epoch)
            for idx, x_i in enumerate(X):
                x_i = Tensor(x_i)
                y_i = Tensor([y_[idx]])
                val = y_i *  (R.dot(x_i, self.w) - self.b)
                condition = R.greater_equal(val,  Scalar(1))
                while condition.status != 'computed':
                    pass
                if condition():
                    self.w = self.w - self.lr * (Scalar(2) * self.lambda_param * self.w)
                else:
                    self.w = self.w - self.lr * (Scalar(2) * self.lambda_param * self.w - R.mul(x_i, y_i))
                    self.b = self.b - (self.lr * y_i)


    def predict(self, X):
        
        X = Tensor(X)
        approx = R.dot(X, self.w) - self.b
        return R.sign(approx)