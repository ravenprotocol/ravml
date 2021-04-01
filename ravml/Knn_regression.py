import ravop.core as R

from ravcom import ravcom
from ravml import metrics


class KNN_Regressor():
    def __init__(self, id=None, **kwargs):
        # super().__init__(id=id, **kwargs)
        self.k = None
        self.n = None
        self.X_train = None
        self.Y = None
        self._label = None
        self._X = None

    def __eucledian_distance(self, X):
        X = R.expand_dims(X, axis=1, name="expand_dims")
        return R.square_root(R.sub(X, self.X_train).pow(R.Scalar(2)).sum(axis=2))

    def fit(self, X_train, Y_train, n_neighbours=None):
        self.k = n_neighbours
        self.n = len(X_train)
        self.X_train = R.Tensor(X_train, name="X_train")
        self.Y = R.Tensor(Y_train, name="Y")
        pass

    def predict(self, X):
        n_q = len(X)
        self._X = R.Tensor(X)
        d_list = self.__eucledian_distance(self._X)
        # print(d_list)
        fe = d_list.foreach(operation='sort')
        sl = fe.foreach(operation='slice', begin=0, size=self.k)
        while sl.status != "computed":
            pass
        pred = R.Tensor([], name="prediction")
        for i in range(n_q):
            row = R.gather(d_list, R.Tensor([i])).reshape(shape=[self.n])

            values = sl.gather(R.Tensor([i])).reshape(shape=[self.k])
            while values.status != 'computed':
                pass
            ind = R.find_indices(row, values)
            while ind.status != 'computed':
                pass
            ind = ind.foreach(operation='slice', begin=0, size=1)
            y_neighbours = R.gather(self.Y, ind).reshape(shape=[self.k])
            while y_neighbours.status != 'computed':
                pass
            pred = pred.concat(R.mean(y_neighbours).expand_dims(axis=0))
            while pred.status != 'computed':
                pass
            print(pred)

        while pred.status != 'computed':
            pass
        self._label = pred
        return pred

    def score(self, y_test):
        return metrics.r2_score(y_test, self._label)

    @property
    def label(self):
        if self._label is None:
            self._label = ravcom.get_ops_by_name(op_name="label", graph_id=self.id)[0]
        print(self._label.id)

        if self._label.status == "computed":
            return self._label.output
        else:
            raise Exception("Need to complete the prediction first")

    @property
    def points(self):
        if self._X is None:
            self._X = ravcom.get_ops_by_name(op_name="X_train", graph_id=self.id)[0]
        print(self._label.id)

        if self._X.status == "computed":
            return self._X.output
        else:
            raise Exception("Need to complete the prediction first")

    def __str__(self):
        return "KNearestNeighboursClassifier:Graph Id:{}\n".format(self.id)
