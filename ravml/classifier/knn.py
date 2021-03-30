import ravop.core as R
from ravop.core import Tensor,Scalar,Graph
from ravcom.utils import inform_server
import numpy as np
def eucledian_distance(self, X, Y):
    return R.square_root(((R.sub(X, Y)).pow(Scalar(2))).sum(axis=0))

'''
            Regressor
'''
class KNN_regressor():

    def __init__(self):
        self.k=None
        self.X=None
        self.y=None
        self.weights=None
        pass

    def fit(self,X_Train,Y_train):
        self.X=Tensor(X_Train)
        self.Y=Tensor(Y_train)
        pass


'''
                    classifier
'''


class KNN_classifier():
    def __init__(self,X_train,Y_train,n_neighbours=None,n_classes=None):
        self.k = n_neighbours
        self.n_c= n_classes
        self.n=len(X_train)
        self.X_train=Tensor(X_train,name = "X")
        self.Y=Tensor(Y_train,name="Y")
        pass

    def eucledian_distance(self,X):
        X = R.expand_dims(X, axis=1,name="expand_dims")
        return R.square_root(R.sub(X, self.X_train).pow(Scalar(2)).sum(axis=2))

    def predict(self, X):
        self.n_q=len(X)
        self.X = Tensor(X)
        d_list=self.eucledian_distance(self.X)
        #print(d_list)
        fe=d_list.foreach(operation='sort')
        sl= fe.foreach(operation='slice',begin=0,size=self.k)
        while sl.status != "computed":
            pass

        print(sl)

        label=R.Tensor([])
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
        return label

