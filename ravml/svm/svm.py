import ravop.core as R
from ravop.core import Scalar, Tensor, Graph

class SVM:
    def __init__(self, learning_rate = 0.001, lambda_param = 0.01, n_iters = 1000):

        self.lr = Scalar(learning_rate)
        self.lambda_param = Scalar(lambda_param)
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):

        # X = Tensor(X)
        # Y = Tensor(Y)

        n_samples, n_features = X.shape
        y_ = R.where(y <= 0, -1, 1)
        self.w = R.zeros(n_features)
        self.b = Scalar(0)

        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                x_i = Tensor(x_i)
                y_i = Scalar(y[idx])
                condition = y_i *  (R.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w = self.w - self.lr * (Scalar(2) * self.lambda_param * self.w)
                else:
                    self.w = self.w - self.lr * (Scalar(2) * self.lambda_param * self.w - R.dot(x_i, y_i))
                    self.b = self.b - (self.lr * y_i)


    def   predict(self, X):
        X = Tensor(X)
        approx = R.dot(X, self.w) - self.b

        return R.sign(approx)