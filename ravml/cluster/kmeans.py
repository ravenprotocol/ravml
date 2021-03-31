import ravop.core as R
from ravop.core import Graph, Tensor, Scalar, square_root,add, min
#from ravop.utils import inform_server
import matplotlib.pyplot as plt
import ravcom

class Kmeans(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self._points = None
        self._label = None
        self.centroids=None
        self.k = None

    def set_params(self, **kwargs):
        self.params.update(**kwargs)

    def get_params(self):
        return self.params

    def fit(self, X, k=3, iter=10):
        self._points = R.Tensor(X,name="points")
        self.k = k
        self.centroids = self.initialize_centroids()
        #inform_server()
        self._label = self.closest_centroids(self.centroids)
        self.update_centroids()

        for i in range(iter):
            print('iteration',i)
            self.update_centroids()
            self._label = self.closest_centroids(self.centroids)
            #inform_server()
        while self._label.status!="computed":
            pass

    def initialize_centroids(self):
        return R.random(self._points, size=self.k)

    def closest_centroids(self, centroids):
        centroids = R.expand_dims(centroids, axis=1)
        return R.argmin(square_root(R.sub(self._points, centroids).pow(Scalar(2)).sum(axis=2)))


    def update_centroids(self):

        gather = self._points.gather(R.find_indices(self._label, Tensor([0]))).mean(axis=1)
        for i in range(1, self.k):
            ind = R.find_indices(self._label, Tensor([i]))
            gat = R.gather(self._points, ind).mean(axis=1)
            gather = R.concat(gather, gat)
        self.centroids= gather.reshape(shape=[self.k, len(self._points.output[0])])
        #inform_server()
    def plot(self):
        fig, axs = plt.subplots(1)
        axs.scatter(self._points.output[:, 0], self._points.output[:, 1], c=self._label.output)
        #axs.scatter(self.centroids.output[:, 0], self.centroids.output[:, 1] ,'X', color="black",markersize=10)
        plt.show()

    @property
    def Points(self):
        if self._points is None:
            self._points = ravcom.get_ops_by_name(op_name="points", graph_id=self.id)[0]
        print(self._points.id)

        if self._points.status == "computed":
            return self._points.output
        else:
            raise Exception("Need to complete the prediction first")

    @property
    def label(self):
        if self._label is None:
            self._label = ravcom.get_ops_by_name(op_name="label", graph_id=self.id)[0]
        print(self._label.id)

        if self._label.status == "computed":
            return self._label.output
        else:
            raise Exception("Need to complete the prediction first")


class MiniBatchKmeans(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.X=None
        self._points = None
        self._label = None
        self.centroids = None
        self.batch_size=None
        self.k = None

    def set_params(self, **kwargs):
        self.params.update(**kwargs)

    def get_params(self):
        return self.params

    def initialize_centroids(self):
        cen=R.random(self._points, size=self.k)
        while cen.status!='computed':
            pass
        return cen

    def Mini_batch(self,points,batch_size):
        mb=R.random(points,size=batch_size)
        return mb

    def closest_centroids(self,points,centroids):
        centroids = R.expand_dims(centroids, axis=1)
        return R.argmin(R.square_root(R.sum(R.square(R.sub(points, centroids)), axis=2)))


    def update_centroids(self,points,label):
        while label.status!= 'computed':
            pass
        if 0 in label.output :
            gather=R.gather(points,R.find_indices(label,Tensor([0]))).mean(axis=1)
        else:
            gather=R.gather(self.centroids, Tensor([0])).expand_dims(axis=0)

        for i in range(1,self.k):
            if i in label.output:
                ind = R.find_indices(label,Tensor([i]))
                gat = R.gather(points,ind).mean(axis=1)
            else:
                gat = R.gather(self.centroids, Tensor([i])).expand_dims(axis=0)
            gather=R.concat(gather,gat)

            while gat.status!='computed':
                pass
        return gather.reshape(shape=[self.k,len(self._points.output[0])])


    def fit(self, X, k , iter=5,batch_size=None):
        #inform_server()
        self._points=Tensor(X)
        self.k=k
        self.iter=iter
        self.batch_size=batch_size
        self.centroids=self.initialize_centroids()
        #self._label=self.closest_centroids(self.points,self.centroids)
        points=self.Mini_batch(self._points,batch_size=batch_size)
        _label=self.closest_centroids(points,self.centroids)
        print(3)
        self.centroids=self.update_centroids(points,_label)
        #inform_server()
        for i in range(iter):
            print('iteration',i)
            points = self.Mini_batch(self._points, batch_size=self.batch_size)
            _label = self.closest_centroids(points, self.centroids)
            self.centroids=self.update_centroids(points,_label)

            #inform_server()

        self._label = self.closest_centroids(self._points, self.centroids)
        while self._label.status!="computed":
            pass
        return self._label
    def plot(self):
        fig, axs = plt.subplots(1)
        axs.scatter(self._points.output[:, 0], self._points.output[:, 1], c=self._label.output)
        # axs.scatter(self.centroids.output[:, 0], self.centroids.output[:, 1] ,'X', color="black",markersize=10)
        plt.show()


    @property
    def points(self):
        if self._points is None:
            self._points = ravcom.get_ops_by_name(op_name="points", graph_id=self.id)[0]
        print(self._points.id)

        if self._points.status == "computed":
            return self._points.output
        else:
            raise Exception("Need to complete the prediction first")

    @property
    def label(self):
        if self._label is None:
            self._label = ravcom.get_ops_by_name(op_name="label", graph_id=self.id)[0]
        print(self._label.id)

        if self._label.status == "computed":
            return self._label.output
        else:
            raise Exception("Need to complete the prediction first")




if __name__ == '__main__':
    pass

