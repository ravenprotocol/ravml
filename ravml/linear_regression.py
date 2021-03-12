import logging
import logging.handlers

import numpy as np

from ravop import globals as g
from ravop.core import Graph, Tensor, Scalar
from ravml import metrics
import ravop.core as R


class LinearRegression(Graph):
    def __init__(self, id=None, **kwargs):
        super().__init__(id=id, **kwargs)

        self.__setup_logger()

        # Define hyper-parameters
        self.learning_rate = R.Scalar(kwargs.get("learning_rate", 0.01), name="learning_rate")
        self.iterations = kwargs.get("iterations", 100)

        self.X = None
        self.y = None
        self.W = None
        self.b = None
        self.no_samples = None
        self.no_features = None

    def __setup_logger(self):
        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(LinearRegression.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(g.ravop_log_file)

        self.logger.addHandler(handler)

    def train(self, X, y):
        # Convert input values to RavOp tensors
        self.X = Tensor(X, name="X")
        self.y = Tensor(y, name="y")

        # Initialize params
        self.no_features = Scalar(X.shape[1], name="no_features")
        self.no_samples = Scalar(X.shape[0], name="no_samples")
        self.W = Tensor(np.zeros((self.no_features.output, 1)), name="W")
        self.b = Scalar(0, name="b")
        # self.weights = Tensor(np.random.uniform(0, 1, self.no_features).reshape((self.no_features, 1)), name="weights")

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()

        return self

        # 1. Predict
        y_pred = X.matmul(weights, name="y_pred")

        # 2. Compute cost
        cost = self.__compute_cost(y, y_pred, no_samples)

        # 3. Gradient descent - Update weight values
        for i in range(iter):
            y_pred = X.matmul(weights, name="y_pred{}".format(i))
            c = X.transpose().matmul(y_pred)
            d = self.learning_rate.div(no_samples)
            weights = weights.sub(c.multiply(d), name="weights{}".format(i))
            cost = self.__compute_cost(y, y_pred, no_samples, name="cost{}".format(i))

        return cost, weights

    def predict(self, X, weights=None):
        """Predict values"""
        return R.matmul(X, self.weights).add(self.bias)

    def update_weights(self):
        y_pred = self.predict(self.X)

        dW = Scalar(-1).multiply(Scalar(2).multiply(self.X.transpose().dot(self.y.sub(y_pred))).div(self.no_samples))
        db = Scalar(-2).multiply(R.sum(self.y.sub(y_pred))).div(self.no_samples)

        self.W = self.W.sub(self.learning_rate.multiply(dW), name="W")
        self.b = self.b.sub(self.learning_rate.multiply(db), name="b")

        return self

    def __compute_cost(self, y, y_pred, no_samples, name="cost"):
        """Cost function"""
        return R.multiply(R.Scalar(1.0/(2.0*no_samples.output)), R.sum(R.square(R.sub(y_pred, y))), name=name)
        # a = y_pred.sub(y)
        # b = R.square(a).sum()
        # R.one()
        # cost = R.one().div(Scalar(2).multiply(no_samples)).multiply(b, name=name)
        # return cost

    @property
    def weights(self):
        """Retrieve weights"""
        if self.W is not None:
            return self.W

        ops = self.get_ops_by_name(op_name="W", graph_id=self.id)
        if len(ops) == 0:
            raise Exception("You need to train your model first")

        # Get weights
        weight_op = ops[-1]
        if weight_op.status == "pending" or weight_op.status == "computing":
            raise Exception("Please wait. Your model is getting trained")

        return weight_op

    @property
    def bias(self):
        """Retrieve bias"""
        if self.b is not None:
            return self.b

        ops = self.get_ops_by_name(op_name="b", graph_id=self.id)
        if len(ops) == 0:
            raise Exception("You need to train your model first")

        # Get weights
        b_op = ops[-1]
        if b_op.status == "pending" or b_op.status == "computing":
            raise Exception("Please wait. Your model is getting trained")

        return b_op

    def score(self, X, y, name="r2"):
        g.graph_id = None
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

    def __str__(self):
        return "LinearRegression:Graph Id:{}\n".format(self.id)
