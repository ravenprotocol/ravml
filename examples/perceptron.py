from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import ravop.core as R
from ravml.linear.perceptron import Perceptron

iris_data = load_iris()
 
x_ = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

sc = StandardScaler()
x_ = sc.fit_transform(x_)

encoder = OneHotEncoder(sparse=False)
y_ = encoder.fit_transform(y_)

X_train, X_test, y_train, y_test = train_test_split(x_, y_, shuffle=True, test_size=0.3)

model = Perceptron(input_dims=4, hidden_dims=10, output_dims=3)
model.fit(X_train, y_train, alpha = 0.01, epoch = 3)

pr = model.predict(X_test[1])
print('Prediction : ',pr)

model.plot_metrics()