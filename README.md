# RavML
Ravenverse's Machine Learning Library

RavML is the machine learning library based on RavOp. It contains implementations of various machine learning algorithms like ordinary least squares, linear regression, logistic regression, KNN, Kmeans, Mini batch Kmeans, Decision Tree classifier, and Naive Bayes classifier


[Github](https://github.com/ravenprotocol/ravml.git)

## Installation

Create a virtual environment
    
    virtualenv ravml -p python3
    
Activate the virtual environment
    
    source ravml/bin/activate

Install Dependencies

    pip install git+https://github.com/ravenprotocol/ravml.git

Install Ravml

    python3 setup.py install
 
> **Note :** Use `python3 setup --help` for argument details

---
## RavML Supported Algorithms

### K-Nearest Neighbours
K-Nearest Neighbors (KNN) is a simple machine learning technique for regression and classification problems. KNN algorithms take data and apply similarity metrics to classify fresh data points (e.g. distance function). A majority vote of its neighbours is used to classify it. The information is assigned to the class with the most neighbours. As the number of nearest neighbours grows, so does the value of k, and so does the accuracy.

```python
from ravml.classifier import KNNClassifier

knn = KNNClassifier()
knn.fit(X_train, y_train, n_neighbours=5, n_classes=3)
print(knn.predict(X_test))
print(knn.score(y_test=y_test))
```
You can view a sample implementation of KNN on ravml [here](https://github.com/ravenprotocol/ravml/blob/main/examples/knn_classifier.py)


### Naive Bayes Classifier
It's a classification method based on Bayes' Theorem and the assumption of predictor independence. A Naive Bayes classifier, in simple terms, assumes that the existence of one feature in a class is unrelated to the presence of any other feature. The Naive Bayes model is simple to construct and is especially good for huge data sets. Naive Bayes is renowned to outperform even the most advanced classification systems due to its simplicity.

```python
from ravml.linear_model.naive_bayes import NaiveBayesClassifier

model = NaiveBayesClassifier()

model.fit(X_train, y_train)
y_preds = model.predict(X_test)

calc_preds = []
for y_pred in y_preds:
    keys = list(y_pred.keys())
    calc_pred = {key: y_pred[key]() for key in keys}
    calc_preds.append(calc_pred)

MAPs = []
for pred in calc_preds:
    MAP = max(pred, key= pred.get)
    MAPs.append(MAP)

print("NaiveBayesClassifier accuracy: {0:.3f}".format(model.accuracy(y_test, MAPs)))
```
You can view a sample implementation of Naive Bayes on ravml [here](https://github.com/ravenprotocol/ravml/blob/main/examples/naive_bayes_classifier.py)

### Support Vector Machine
SVM (Support Vector Machine) is a supervised machine learning technique that can be used to solve classification and regression problems. It is, however, mostly employed to solve categorization difficulties. Each data item is plotted as a point in n-dimensional space (where n is the number of features you have), with the value of each feature being the value of a certain coordinate in the SVM algorithm. Then we accomplish classification by locating the hyper-plane that clearly distinguishes the two classes.
### K-Means

Clustering is a type of unsupervised learning wherein data points are grouped into different sets based on their degree of similarity. The k-means clustering algorithm assigns data points to categories, or clusters, by finding the mean distance between data points. It then iterates through this technique in order to perform more accurate classifications over time. Since you must first start by classifying your data into k categories, it is essential that you understand your data well enough to do this

```python
from ravml.cluster import KMeans

k = KMeans()
k.fit(X_train, 3, iter=30)
```

You can view a sample implementation of K-means on ravml [here](https://github.com/ravenprotocol/ravml/blob/main/examples/kmeans.py)

### Linear Regression

Linear Regression is a supervised machine learning technique with a continuous and constant slope projected output. Rather than aiming to classify data into categories (e.g. cat, dog), it is used to predict values within a continuous range (e.g. sales, price). There are two main types: Simple and Multivariable Regression.

```python
from ravml.linear.linear_regression import LinearRegression

model = LinearRegression(x,y,theta)

model.compute_cost()  # initial cost with coefficients at zero

optimal_theta = model.gradient_descent(alpha=0.01, iterations=20)

model.plot_graph(optimal_theta)
```

You can view the implementation of Linear Regression [*here*](https://github.com/ravenprotocol/ravml/blob/main/ravml/linear/linear_regression.py).

### Logistic Regression

The Supervised Learning methodology of logistic regression is used to predict the categorical dependent variable using a set of independent factors. A categorical dependent variable's output is predicted using logistic regression. As a result, the result must be a discrete or categorical value. Logistic Regression is much similar to Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems. 

```python
from ravml.linear.logistic_regression import LogisticRegression
model = LogisticRegression(lr=0.1, num_iter=30)

model.fit(X, y)

preds = model.predict(X)

print((preds == y).mean())
print(model.theta())

model.plot_loss()

model.visualize(X,y)
```

You can view the implementation of Logistic Regression on IRIS flower dataset [*here*](https://github.com/ravenprotocol/ravml/tree/main/examples/logistic_regression.py).

### Multi-Layer Perceptron

The multi-layer perceptron (MLP) is a feed-forward neural network supplement. It has three layers: an input layer, an output layer, and a hidden layer. The input signal to be processed is received by the input layer. The output layer is responsible for tasks such as prediction and categorization. The true computational engine of the MLP is an arbitrary number of hidden layers inserted between the input and output layers. In an MLP, data flows from input to output layer in the forward direction, similar to a feed-forward network. The backpropagation learning algorithm is used to train the neurons in the MLP. MLPs can tackle issues that aren't linearly separable and are designed to approximate any continuous function. Pattern categorization, recognition, prediction, and approximation are some of MLP's most common applications.

```python
from ravml.linear.perceptron import Perceptron

model = Perceptron(input_dims=4, hidden_dims=10, output_dims=3)

model.fit(X_train, y_train, alpha = 0.01, epoch = 3)

pr = model.predict(X_test[1])
print('Prediction : ',pr)

model.plot_metrics()      # plots accuracy and loss
```

You can view the implementation of MLP on IRIS Flower Dataset [*here*](https://github.com/ravenprotocol/ravml/blob/main/ravml/linear/mlp_iris.py).

### Decision Trees

Decision Tree is a supervised learning technique that may be used to solve both classification and regression problems, however, it is most commonly employed to solve classification issues. Internal nodes represent dataset attributes, branches represent decision rules, and each leaf node provides the conclusion in this tree-structured classifier. The Decision Node and the Leaf Node are the two nodes of a Decision tree. Leaf nodes are the output of those decisions and do not contain any more branches, whereas Decision nodes are used to make any decision and have several branches. The decisions or tests are made based on the characteristics of the given dataset.

```python
from ravml.tree import DecisionTreeClassifier

obj = DecisionTreeClassifier(max_depth=3)
obj.fit(X_train[:30], y_train[:30])
pr = obj.predict(X_test)

print(f1_score(y_test, pr, average='weighted'))
```

You can view the implementation of Decision Tree [*here*](https://github.com/ravenprotocol/ravml/blob/main/ravml/tree/decision_tree.py).
