import numpy as np
import matplotlib.pyplot as plt
import ravop.core as R
from ravcom import inform_server

#----------------------------------------------------------
inform_server()
#----------------------------------------------------------

class Perceptron(R.Graph):
    def __init__(self,input_dims,hidden_dims,output_dims):
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.x = None
        self.y = None 
        self.w1 = self.generate_wt(self.input_dims, self.hidden_dims)
        self.w2 = self.generate_wt(self.hidden_dims, self.output_dims)
        self.acc = []
        self.losses = []
        self.leny = 0

    def preprocess(self,X_train,y_train):
        x_ = X_train.tolist()
        y_ = y_train.tolist()
        y = []
        for element in y_:
            t = []
            for e in element:
                t.append(int(e))
            y.append(R.Tensor(t))

        x = []
        for element in x_:
            x.append(R.Tensor(element).reshape(shape=[1,self.input_dims]))

        return x,y

    def sigmoid(self,n):
        # x is already a R.Scalar or a R.Tensor
        return (R.Scalar(1).div(R.Scalar(1).add(R.exp(n.multiply(R.Scalar(-1))))))
        
    # Initializing the weights randomly
    def generate_wt(self,m, n):
        l =[]
        for i in range(m * n):
            l.append(np.random.randn())
        return(R.Tensor(l).reshape(shape=[m,n]))

    def loss(self,out, Y):
        # out and Y are both R.Tensor objects
        s = R.square(out.sub(Y))
        s = R.sum(s).div(R.Scalar(self.leny))
        return(s)

    def f_forward(self,x, w1, w2):
        # hidden
        z1 = x.dot(w1)# input from layer 1 
        a1 = self.sigmoid(z1)# out put of layer 2 
        
        # Output layer
        z2 = a1.dot(w2)# input of out layer
        a2 = self.sigmoid(z2)# output of out layer
        
        return a2

    # Back propagation of error. 
    def back_prop(self,x, y, w1, w2, alpha):
        # x,y are Tensors  
        # hiden layer
        z1 = x.dot(w1)# input from layer 1 
        a1 = self.sigmoid(z1)# output of layer 2 
        # Output layer
        z2 = a1.dot(w2)# input of out layer
        a2 = self.sigmoid(z2)# output of out layer
        # error in output layer
        d2 = a2.sub(y)
        d3 = a1.multiply(R.Scalar(1).sub(a1))
        d4 = w2.dot(d2.transpose()).transpose()
        d1 = d3.multiply(d4)

        # Gradient for w1 and w2
        w1_adj = x.transpose().dot(d1)
        w2_adj = a1.transpose().dot(d2)
        
        # Updating parameters
        w1 = w1.sub(R.Scalar(alpha).multiply(w1_adj))
        w2 = w2.sub(R.Scalar(alpha).multiply(w2_adj))
        
        return(w1, w2)

    def fit(self, X_train, y_train, alpha = 0.01, epoch = 1):
        self.x,self.y = self.preprocess(X_train,y_train)
        self.leny = len(self.y)
        print('Starting to Train...')
        for j in range(epoch):
            loss_sum = R.Scalar(0)
            for i in range(len(self.x)):
                out = self.f_forward(self.x[i], self.w1, self.w2)  
                loss_sum = loss_sum.add(self.loss(out, self.y[i]))
                self.w1, self.w2 = self.back_prop(self.x[i], self.y[i], self.w1, self.w2, alpha)
            print('Epoch : ',j+1)
            self.acc.append(R.Scalar(1).sub(loss_sum.div(R.Scalar(len(self.x)))).multiply(R.Scalar(100)))
            self.losses.append(loss_sum.div(R.Scalar(len(self.x))))
        while self.w1.status!='computed':
            pass
        while self.w2.status!='computed':
            pass
        print('Training Complete!')

    def predict(self, X_test):
        x_test = X_test.tolist()
        test = R.Tensor(x_test).reshape(shape=[1,self.input_dims])
        lastout = self.f_forward(test, self.w1, self.w2)
        maxm = 0
        k = 0
        while lastout.status != 'computed':
            pass
        temp = list(lastout()[0])
        k = temp.index(max(temp))
        return k

    def plot_metrics(self):
        training_accuracy = []
        for element in self.acc:
            while element.status!='computed':
                pass
            training_accuracy.append(element())

        training_loss = []
        for element in self.losses:
            while element.status!='computed':
                pass
            training_loss.append(element())

        # ploting accuraccy
        plt.plot(training_accuracy)
        plt.ylabel('Accuracy')
        plt.xlabel("Epochs:")
        plt.show()

        plt.clf()

        # plotting Loss
        plt.plot(training_loss)
        plt.ylabel('Loss')
        plt.xlabel("Epochs:")
        plt.show()