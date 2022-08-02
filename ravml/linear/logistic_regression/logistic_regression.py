import numpy as np
import ravop.core as R


class LogisticRegression():
    def __init__(self, lr=0.01, num_iter=10, fit_intercept=False, verbose=True):
        self.lr = R.Scalar(lr)
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.x_shape1 = None
        self.losses = []
        self.preds = None

    def __add_intercept(self, X):
        self.x_shape1 = X.shape[1] + 1
        intercept = np.ones((X.shape[0], 1))
        temp = np.concatenate((intercept, X), axis=1)
        return R.Tensor(temp)

    def __sigmoid(self, z):
        return (R.Scalar(1).div(R.Scalar(1).add(R.exp(z.multiply(R.Scalar(-1))))))

    def __loss(self, h, y):
        c1 = R.Scalar(-1).multiply(y.multiply(h.natlog()))
        c2 = (R.Scalar(1).sub(y)).multiply((R.Scalar(1).sub(h)).natlog())
        c3 = c1.sub(c2)
        c4 = c3.sum().div(R.Scalar(self.leny))
        return c4

    def fit(self, X, y):
        self.leny = len(y)
        Y = R.Tensor(y)
        self.theta = R.Tensor([0] * np.shape(X)[1])
        X = R.Tensor(X)
        for i in range(self.num_iter):
            h = self.__sigmoid(X.dot(self.theta))
            w = X.transpose()
            self.theta = self.theta.sub(self.lr.multiply((w.dot(h.sub(Y)).div(R.Scalar(self.leny)))))
            loss = self.__loss(self.__sigmoid(X.dot(self.theta)), Y)
            print('Iteration : ', i)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(X.dot(self.theta))

    def predict(self, X):
        p = self.predict_prob(R.t(X))
        p.persist_op(name="predicted_vals")
        return p
