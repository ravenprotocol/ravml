import imp
import ravop.core as R
import numpy as np
import ravml.metrics as m

from ravml.kernels import Kernels

class SVM_classifier():
    def __init__(self,
        learning_rate=0.01,
        lamda_param=1,
        gamma=0.2,
        C=None,
        coefficeint=None,
        degree=0,
        n_iterations=10,
        kernel='rbf'):
            self.lr = R.Scalar(learning_rate)
            self.lambda_param = lamda_param
            self.w = None
            self.b = None
            self.kernel=Kernels[kernel]
            self.n_iters=n_iterations
            self.gamma=gamma
            self.coef=coefficeint

            self.trained= False
            self.y_pred=None
            self.w=None

    def fit(self,X,y,sample_weight=None):
        X=R.Tensor(X)
        _x_output=X()
        y=R.Tensor(y)
        shape_x=R.shape(X)()
        n_samples=shape_x[0]
        n_features=shape_x[1]
        # kernel function
        kernel_func=self.kernel
        #kernel matrix or gram matrix
        kernel_matrix=np.zeros((n_samples,n_samples))
        print("==>",kernel_matrix,"\n\n")
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i][j]=kernel_func(X1=R.t(_x_output[i]),X2=R.t(_x_output[j]),gamma=self.gamma)

        self.K=kernel_matrix
        print("gram matrix for calculation:\n",self.K)
        


        # required parameter for matrix solver :
        P=R.t(self.k).multiply(R.t(np.outer(y(),y())))
        q=-1*np.ones(n_samples)
        A=0
        b=0

        g_max=np.identity(n_samples)
        g_min=np.identity(n_samples) * -1

        # solve the quadratic matrix equation problem :
        for _ in range(self.n_iters):
            for idxM in range(len(self.lambda_param)):
                idxL = np.random.randint(0, len(self.lambda_param))
                Q = self.K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                v0 = self.lambda_param[[idxM, idxL]]
                k0 = 1 - np.sum(self.lambda_param * self.K[idxM, idxL], axis=1)
                u = np.array([-self.y[idxL], self.y[idxM]])
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
                self.lambda_param[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)
        # training begin
        for _ in range(self.n_iters):
            pass
            #training takes place here    
            self._train_step()



    def _train_step(self,weight=None):  # temporary name as weight (CHANGE IT!!!!!!)
        pass

    def predict(self,X):
        self.y_pred =10
        if self.trained is False or self.w is None:
            raise Exception("Provide model weights or train using fit() function before predicting with test data")

        pass
    
    def score(self,y_true=None,y_pred=None,metric=m.accuracy):
        if y_pred is None or self.y_pred is None :
            raise Exception("Provide the predicted values or predict using the predict function() before scoring the moddel")

        if y_pred is None:
            y_pred=self.y_pred
        
        val_score=metric(y_true,self.predict(y_pred))

        return val_score()
        
    @property
    def get_params(self):
        #return learnable params
        param_dict={
            'lamda': self.lambda_param(),
            'w':self.w(),
            'coefficients':self.coef(),
            'gamma':self.gamma(),
            'lr':self.lr
        }
        return param_dict

    @property
    def set_params(self,**kwargs):
        #set learnable params
        param_dict={
            'lamda': self.lambda_param(),
            'w':self.w(),
            'coefficients':self.coef(),
            'gamma':self.gamma(),
            'lr':self.lr
        }
        for i in kwargs.keys():
            if i in param_dict.keys():
                param_dict[i]=kwargs[i]

        return param_dict

        











