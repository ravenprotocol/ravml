from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import ravop.core as R
from ravcom import inform_server
import numpy as np              # for random number generator.
import matplotlib.pyplot as plt
import time

#----------------------------------------------------------
inform_server()
#----------------------------------------------------------


iris_data = load_iris()

sc = StandardScaler()
 
x_ = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

sc = StandardScaler()
x_ = sc.fit_transform(x_)

encoder = OneHotEncoder(sparse=False)
y_ = encoder.fit_transform(y_)


X_train, X_test, y_train, y_test = train_test_split(x_, y_, shuffle=True, test_size=0.3)

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
    x.append(R.Tensor(element).reshape(shape=[1,4]))

leny = len(y)


def sigmoid(x):
    # x is already a R.Scalar or a R.Tensor
    return (R.Scalar(1).div(R.Scalar(1).add(R.exp(x.multiply(R.Scalar(-1))))))

def f_forward(x, w1, w2):
    # hidden
    z1 = x.dot(w1)# input from layer 1 
    a1 = sigmoid(z1)# out put of layer 2 
      
    # Output layer
    z2 = a1.dot(w2)# input of out layer
    a2 = sigmoid(z2)# output of out layer
    
    return a2

# Initializing the weights randomly
def generate_wt(x, y):
    l =[]
    for i in range(x * y):
        l.append(np.random.randn())
    return(R.Tensor(l).reshape(shape=[x,y]))

def loss(out, Y):
    # out and Y are both R.Tensor objects
    s = R.square(out.sub(Y))
    s = R.sum(s).div(R.Scalar(leny))
    return(s)

# Back propagation of error. 
def back_prop(x, y, w1, w2, alpha):
    # x,y are Tensors  
    # hiden layer
    z1 = x.dot(w1)# input from layer 1 
    a1 = sigmoid(z1)# output of layer 2 
    # Output layer
    z2 = a1.dot(w2)# input of out layer
    a2 = sigmoid(z2)# output of out layer
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

def train(x, Y, w1, w2, alpha = 0.01, epoch = 10):
    acc = []
    losses = []
    for j in range(epoch):
        loss_sum = R.Scalar(0)
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)  
            loss_sum = loss_sum.add(loss(out, Y[i]))
            w1, w2 = back_prop(x[i], y[i], w1, w2, alpha)
        print('Epoch : ',j+1)
        acc.append(R.Scalar(1).sub(loss_sum.div(R.Scalar(len(x)))).multiply(R.Scalar(100)))
        losses.append(loss_sum.div(R.Scalar(len(x))))
    return(acc, losses, w1, w2)

def predict(x, w1, w2):
    lastout = f_forward(x, w1, w2)
    maxm = 0
    k = 0
    while lastout.status != 'computed':
        pass

    temp = list(lastout()[0])
    k = temp.index(max(temp))
    return k
 
# Initializing weights  
w1 = generate_wt(4, 10)
w2 = generate_wt(10, 3)

# Training the Neural Network
acc, losses, w1, w2 = train(x, y, w1, w2, 0.1, 10)

while w1.status!='computed':
    pass

# Printing trained weights
print(w1())

while w2.status!='computed':
    pass

# Printing trained weights
print(w2())

# Testing data
x_test = X_test.tolist()
test = [R.Tensor(x_test[0]).reshape(shape=[1,4])]

# Making prediction on test data with trained weights
print('Prediction Test : ',end='')
print(test[0])
print('Index of Predicted class : ',end='')
output = predict(test[0], w1, w2)
print(output)

training_accuracy = []
for element in acc:
    while element.status!='computed':
        pass
    training_accuracy.append(element())

print('Training Accuracy : ')
print(training_accuracy)

training_loss = []
for element in losses:
    while element.status!='computed':
        pass
    training_loss.append(element())

print('Training Loss : ')
print(training_loss)


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