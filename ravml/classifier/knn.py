import ravop.core as R
from ravop.core import Tensor,Scalar,Graph
from ravml import metrics
import ravcom



'''
                    KNN classifier
'''


class KNN_classifier():
    def __init__(self, id=None, **kwargs):
        #super().__init__(id=id, **kwargs)
        self.k = None
        self.n_c= None
        self.n=None
        self.X_train=None
        self.Y=None

    def set_params(self, **kwargs):
        self.params.update(**kwargs)

    def get_params(self):
        return self.params

    def __eucledian_distance(self,X):
        X = R.expand_dims(X, axis=1,name="expand_dims")
        return R.square_root(R.sub(X, self.X_train).pow(Scalar(2)).sum(axis=2))

    def fit(self,X_train,Y_train,n_neighbours=None,n_classes=None):
        self.k = n_neighbours
        self.n_c = n_classes
        self.n = len(X_train)
        self.X_train = Tensor(X_train, name="X_train")
        self.Y = Tensor(Y_train, name="Y")
        pass

    def predict(self, X):
        self.n_q=len(X)
        self._X = Tensor(X)
        d_list=self.__eucledian_distance(self._X)
        #print(d_list)
        fe=d_list.foreach(operation='sort')
        sl= fe.foreach(operation='slice',begin=0,size=self.k)
        while sl.status != "computed":
            pass
        label=R.Tensor([],name="label")
        for i in range(self.n_q):
            row=R.gather(d_list,Tensor([i])).reshape(shape=[self.n])

            values=sl.gather(Tensor([i])).reshape(shape=[self.k])
            while values.status!='computed':
                pass
            print(values,row)
            ind=R.find_indices(row,values)

            while ind.status!='computed':
                pass

            ind=ind.foreach(operation='slice',begin=0,size=1)
            y_neighbours= R.gather(self.Y,ind).reshape(shape=[self.k])
            while y_neighbours.status!='computed':
                pass
            label=label.concat(R.mode(y_neighbours))

        while label.status!='computed':
            pass
        self._label=label
        return label

    def score(self,y_test):
        return metrics.r2_score(y_test,self._label)


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
    def Points(self):
        if self._X is None:
            self._X = ravcom.get_ops_by_name(op_name="X_train", graph_id=self.id)[0]
        print(self._label.id)

        if self._X.status == "computed":
            return self._X.output
        else:
            raise Exception("Need to complete the prediction first")

    def __str__(self):
        return "KNearestNeighboursClassifier:Graph Id:{}\n".format(self.id)