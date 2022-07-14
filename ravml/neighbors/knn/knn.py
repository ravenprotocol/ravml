import ravop.core as R
from ravop.core import Tensor, Scalar

from ravml.metrics import metrics

class KNNClassifier():
    def __init__(self, **kwargs):
        self._k = None
        self._n_c = None
        self._n = None
        self._X = None
        self._y = None
        self._labels = None

    def __euclidean_distance(self, X):
        X = R.expand_dims(X, axis=1)
        return R.square_root(R.sub(X, self._X).pow(R.t(2)).sum(axis=2))

    def fit(self, X, y, n_neighbours=None, n_classes=None):
        if n_neighbours is None or n_classes is None:
            raise Exception("Required params: n_neighbours, n_classes")

        self._X = R.t(X)
        self._y = R.t(y)
        self._k = n_neighbours
        self._n_c = n_classes
        self._n = len(X)
       

    def predict(self, X):
        n_q = len(X)
        X = R.Tensor(X)
        d_list = self.__euclidean_distance(X)
        print("calculating euclidian distance...")
        fe = d_list.foreach(operation='sort')
        sl = fe.foreach(operation='slice', begin=0, size=self._k)
        label = R.Tensor([])
        

        for i in range(n_q):
            row = d_list.gather(R.t([i])).squeeze()
            values = sl.gather(R.t([i])).squeeze()
            ind = row.find_indices(values).foreach(operation='slice', begin=0, size=1)
            y_neighbours = R.gather(self._y, ind.reshape(shape=[self._k]))
            label = label.concat(R.mode(y_neighbours))
        label.persist_op(name="label")

        self._labels = label

        # return label

    def score(self, y_test):
        acc = metrics.accuracy(y_test, self._labels)
        acc.persist_op(name="accuracy")
        return acc

    
    @property
    def label(self):
        return self._label

    @property
    def points(self):
        return self._X

    def set_params(self, **kwargs):
        param_dict={
            'labels': self._labels,
            'X':self._X,
            'y':self._y,
            'k':self.k
        }
        for i in kwargs.keys():
            if i in param_dict.keys():
                param_dict[i]=kwargs[i]
        return param_dict

    def get_params(self):
        param_dict={
            'labels': self._labels,
            'X':self._X,
            'y':self._y,
            'k':self.k
        }
        return param_dict




class KNNRegressor():
    def __init__(self, id=None, **kwargs):
        # super().__init__(id=id, **kwargs)
        self.k = None
        self.n = None
        self.X_train = None
        self.Y = None
        self._label = None
        self._X = None

    def __eucledian_distance(self, X):
        X = R.expand_dims(X, axis=1)
        return R.square_root(R.sub(X, self.X_train).pow(R.t(2)).sum(axis=2))

    def fit(self, X_train,Y_train, n_neighbours=None):
        self.k = n_neighbours
        self.n = len(X_train)
        self.X_train = R.t(X_train)
        self._k = n_neighbours

        self._y = R.t(Y_train)
        

    def predict(self, X):
        n_q = len(X)
        self._X = R.t(X)
        d_list = self.__eucledian_distance(self._X)
        fe = d_list.foreach(operation='sort')
        sl = fe.foreach(operation='slice', begin=0, size=self.k)

        pred = R.Tensor([])
        for i in range(n_q):

            row = d_list.gather(R.t([i])).squeeze()
            values = sl.gather(R.t([i])).squeeze()
            ind = row.find_indices(values).foreach(operation='slice', begin=0, size=1)
            y_neighbours = R.gather(self._y, ind.reshape(shape=[self._k]))

            pred = pred.concat(R.mean(y_neighbours).expand_dims(axis=0))
        self._label = pred
        pred.persist_op(name="predicted_label")

    def score(self, y_test):
        score=metrics.r2_score(y_test, self._label)
        score.persist_op(name="r2score_knn_classifier")


    @property
    def label(self):
        return self._label

    @property
    def points(self):
        return self._X

    def set_params(self, **kwargs):
        param_dict={
            'labels': self._labels,
            'X':self._X,
            'y':self._y,
            'k':self.k
        }
        for i in kwargs.keys():
            if i in param_dict.keys():
                param_dict[i]=kwargs[i]
        return param_dict

    def get_params(self):
        param_dict={
            'labels': self._labels,
            'X':self._X,
            'y':self._y,
            'k':self.k
        }
        return param_dict



