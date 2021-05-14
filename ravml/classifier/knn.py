from ravcom import ravcom
import ravop.core as R
from ravop.core import Tensor, Scalar

from ravml import metrics
from ravml.base import Base

'''
KNN classifier
'''


class KNNClassifier(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._k = None
        self._n_c = None
        self._n = None

        self._X = None
        self._y = None

        self._labels = None

    def __euclidean_distance(self, X):
        X = R.expand_dims(X, axis=1, name="expand_dims")
        return R.square_root(R.sub(X, self._X).pow(Scalar(2)).sum(axis=2))

    def fit(self, X, y, n_neighbours=None, n_classes=None):
        if n_neighbours is None or n_classes is None:
            raise Exception("Required params: n_neighbours, n_classes")

        self._X = Tensor(X, name="X_train")
        self._y = Tensor(y, name="y_train")

        # Params
        self._k = n_neighbours
        self._n_c = n_classes
        self._n = R.shape(self._X)
        while self._n.status !='computed':
            pass
        self._n=int(self._n.output[0])


    def predict(self, X):
        n_q = len(X)
        X = Tensor(X)
        d_list = self.__euclidean_distance(X)
        fe = d_list.foreach(operation='sort')
        sl = fe.foreach(operation='slice', begin=0, size=self._k)
        label = R.Tensor([], name="label")

        for i in range(n_q):
            row = d_list.gather(Tensor([i])).reshape(shape=[self._n])
            values = sl.gather(Tensor([i])).reshape(shape=[self._k])
            ind = row.find_indices(values)
            ind = ind.foreach(operation='slice', begin=0, size=1)
            y_neighbours = R.gather(self._y, ind).reshape(shape=[self._k])
            label = label.concat(R.mode(y_neighbours))

        # Store labels locally
        while label.status != 'computed':
            pass
        self._labels = label

        return label

    def score(self, y_test):
        acc= metrics.accuracy(y_test, self._labels)
        while acc.status!='computed':
            pass
        return acc

    @property
    def labels(self):
        if self._labels is None:
            self._labels = ravcom.get_ops_by_name(op_name="label", graph_id=self.id)[0]
        print(self._labels.id)

        if self._labels.status == "computed":
            return self._labels.output
        else:
            raise Exception("Need to complete the prediction first")

    @property
    def points(self):
        if self._X is None:
            self._X = ravcom.get_ops_by_name(op_name="X_train", graph_id=self.id)[0]

        if self._X.status == "computed":
            return self._X.output
        else:
            raise Exception("Need to complete the prediction first")

    def __str__(self):
        return "KNNClassifier"
