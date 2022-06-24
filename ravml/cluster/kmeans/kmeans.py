from cProfile import label
import matplotlib.pyplot as plt
import selenium
import ravop.core as R
from ravop.core import Tensor, Scalar, square_root
import numpy as np

class KMeans():
    def __init__(self, **kwargs):
        self.params = kwargs
        self._points = None
        self._label = None
        self.centroids = None
        self.k = None

    def set_params(self, **kwargs):
        param_dict={
            'label': self._label(),
            'points':self._points(),
            'centroids':self._centroids(),
            'k':self.k
        }
        for i in kwargs.keys():
            if i in param_dict.keys():
                param_dict[i]=kwargs[i]

        return param_dict

        
    def get_params(self):
        param_dict={
            'label': self._label(),
            'points':self._points(),
            'centroids':self._centroids(),
            'k':self.k
        }
        return param_dict

    def fit(self, X, k=3, iter=10):
        self._points = R.t(X)
        self.k = k
        self.centroids = self.initialize_centroids()
        print(self.centroids(),self._points())
        self._label = self.closest_centroids(self.centroids)
        self.update_centroids()

        for i in range(iter):
            print('iteration', i)
            
            self._label = self.closest_centroids(self.centroids)
            self.update_centroids()
            self._label()


    def initialize_centroids(self):
        return R.t(   self._points() [np.random.choice(len(self._points()), size=self.k)] )

    def closest_centroids(self, centroids):
        centroids = R.expand_dims(centroids, axis=1)
        sub=R.sub(self._points, centroids).pow(R.t(2)).square_root()
        
        temp=sub.sum(axis=2)
        
        closest_centroids= R.argmin(temp.transpose() ,axis=1)
        print("labels:\n",closest_centroids())
        return closest_centroids

    def update_centroids(self):
        gather = self._points.gather(R.find_indices(self._label, R.t([0]))).transpose().mean(axis=1)
        print("update 0:",gather())
        # gather=R.t([])
        for i in range(1, self.k):
            ind = R.find_indices(self._label, R.t([i]))
            # print(ind())
            gat = R.gather(self._points, ind).transpose().mean(axis=1)
            gather = R.concat(gather, gat)
            print("update >>:",gather())
        self.centroids = gather.reshape(shape=[self.k, len(self._points()[0])])
        

    def plot(self):
        fig, axs = plt.subplots(1)
        axs.scatter(self._points()[:, 0], self._points()[:, 1], c=self._label())
        plt.show()

    @property
    def Points(self):
        return self.Points

    @property
    def label(self):
        return self.label


class MiniBatchKMeans(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.X = None
        self._points = None
        self._label = None
        self.centroids = None
        self.batch_size = None
        self.k = None

    def set_params(self, **kwargs):
        param_dict={
            'label': self._label(),
            'points':self._points(),
            'centroids':self._centroids(),
            'k':self.k
        }
        for i in kwargs.keys():
            if i in param_dict.keys():
                param_dict[i]=kwargs[i]

        return param_dict

        
    def get_params(self):
        param_dict={
            'label': self._label(),
            'points':self._points(),
            'centroids':self._centroids(),
            'k':self.k
        }
        return param_dict

    def initialize_centroids(self):
        return R.t(   self._points() [np.random.choice(len(self._points()), size=self.k)] )

    def Mini_batch(self, points, batch_size):
        mb = R.t(  points() [np.random.choice(len(points()), size=batch_size)] )
        return mb

    def closest_centroids(self,points, centroids):
        # print(points(),"======",centroids())
        centroids = R.expand_dims(centroids, axis=1)
        sub=R.sub(points, centroids).pow(R.t(2)).square_root()
        
        temp=sub.sum(axis=2)
        
        closest_centroids= R.argmin(temp.transpose() ,axis=1)
        print("labels:\n",closest_centroids())
        return closest_centroids

    def update_centroids(self,points,_label):
        
        if 0 in _label():
            ind_i=R.find_indices(_label, R.t([0]))
            gather = points.gather(ind_i).transpose().mean(axis=1)
        else:
            gather= R.t(self.centroids()[0])
            # print("update 0:",gather())

        for i in range(1, self.k):
            
            if i in _label():
                ind = R.find_indices(_label, R.t([i]))
                gat = R.gather(points, ind).transpose().mean(axis=1)
                gather = R.concat(gather, gat)
                # print("update >>:",gather())
            else:
                gat= R.t(self.centroids()[i])
                gather = R.concat(gather, gat)
        self.centroids = gather.reshape(shape=[self.k,len(self._points()[0])])

    def fit(self, X, k, iter=5, batch_size=30):
        # inform_server()
        self._points = R.t(X)
        self.k = k
        self.centroids = self.initialize_centroids()

        self.iter = iter
        self.batch_size = batch_size
        self.centroids = self.initialize_centroids()
        print(self.centroids())
        # self._label=self.closest_centroids(self.points,self.centroids)
        points = self.Mini_batch(self._points, batch_size=batch_size)
        _label = self.closest_centroids(points, self.centroids)
        print(3)
        self.update_centroids(points, _label)
        for i in range(iter):
            print('iteration', i,"\n")
            points = self.Mini_batch(self._points, batch_size=self.batch_size)
            _label = self.closest_centroids(points, self.centroids)
            self.update_centroids(points, _label)
            self.centroids()
            

        self._label = self.closest_centroids(self._points, self.centroids)
        self._label()
        return self._label

    def plot(self):
        fig, axs = plt.subplots(1)
        axs.scatter(self._points()[:, 0], self._points()[:, 1], c=self._label())
        # axs.scatter(self.centroids.output[:, 0], self.centroids.output[:, 1] ,'X', color="black",markersize=10)
        plt.show()

    @property
    def points(self):
        pass

    @property
    def label(self):
        pass
