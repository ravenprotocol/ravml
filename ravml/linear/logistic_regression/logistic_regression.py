import ravop.core as R
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self,lr=0.01, num_iter=10, fit_intercept=True, verbose=True):
        self.lr = R.Scalar(lr)
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.x_shape1 = None
        self.losses = []
        self.preds = None

    def __add_intercept(self, X):
        self.x_shape1 = X.shape[1]+1
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
        if self.fit_intercept:
            X = self.__add_intercept(X)
        self.leny = len(y)
        Y = R.Tensor(y)
        # weights initialization
        self.theta = R.Tensor([0]*self.x_shape1)

        for i in range(self.num_iter):
            h = self.__sigmoid(X.dot(self.theta))
            while h.status != 'computed':
                pass
            w = X.transpose()
            while w.status != 'computed':
                pass
            self.theta = self.theta.sub(self.lr.multiply((w.dot(h.sub(Y)).div(R.Scalar(self.leny)))))
            while self.theta.status!='computed':
                pass
            loss = self.__loss(self.__sigmoid(X.dot(self.theta)), Y)
            while loss.status!='computed':
                pass
            if (self.verbose == True):
                self.losses.append(loss)
            print('Iteration : ',i)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(X.dot(self.theta))

    def predict(self, X):
        p = self.predict_prob(X)
        while p.status!='computed':
            pass
        t = p()
        return t.round()

    def plot_loss(self):
        training_loss = []
        for element in self.losses:
            while element.status != 'computed':
                pass
            training_loss.append(element())
        # plotting Loss
        plt.plot(training_loss)
        plt.ylabel('Loss')
        plt.xlabel("Iteration")
        plt.show()

    def visualize(self,X,y):
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
        plt.legend()
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        self.probs = self.predict_prob(grid) 
        while self.probs.status != 'computed':
            pass
        self.probs = np.array(self.probs())
        self.probs = self.probs.reshape(xx1.shape)
        plt.contour(xx1, xx2, self.probs, [0.5], linewidths=1, colors='black')
        plt.show()