<center>

# <B> K-Nearest Neighbours </B>
</center>

K-Nearest Neighbours (KNN) is a simple machine learning technique for regression and classification problems. KNN algorithms take data and apply similarity metrics to classify fresh data points (e.g. distance function). A majority vote of its neighbours is used to classify it. The information is assigned to the class with the most neighbours. As the number of nearest neighbours grows, so does the value of k, and so does the accuracy.



<center>

# <B> K-Nearest Neighbour Classification </B>
</center>


```python
from ravml.classifier import KNNClassifier

knn = KNNClassifier()
knn.fit(X_train, y_train, n_neighbours=5, n_classes=3)
print(knn.predict(X_test))
print(knn.score(y_test=y_test))
```


<B><center>
## Methods
</center>
</B>

- ### <B><U>fit( X, y, n_neighbours=None, n_classes=None)</u></B>

>
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X {dtype:list/numpy_ndarray,shape:(n_samples,n_features)} | array like matrix denoting the input array  | 
>|    n_neighbours {dtype: int } | Number of neighbours for knn algorithm    |  
>|    n_classes{dtype:int}  | number of classes  for the classification |



- ### <B><U>predict( X)</u></B>

>
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X {dtype:list/numpy_ndarray,shape:(n_samples,n_features)} | array like matrix to predict.  | 


- ### <B><U> score(y_test)</u></B>

>returns the r2 score of the regressor based on the predicted y_test values from the classifier. It uses r2score() metric function available in ravml library.
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    y_test {dtype:list/numpy_ndarray,shape:(n_samples)} | get metrics to evaluate our predictions from the model. | 


- ### <B><U>points()</u></B>

>returns the input X array given to the fit() function
>

- ### <B><U>labels()</u></B>

>returns the labels returned by the model after the model generates them.
>


You can view a sample implementation of KNN on ravml [here](https://github.com/ravenprotocol/ravml/blob/main/examples/knn_classifier.py)



<center>

# <B> K-Nearest Neighbour Regression</B>
</center>

```python
from ravml.classifier import KNNRegressor

knn = KNNRegressor()
knn.fit(X_train, y_train, n_neighbours=5)
print(knn.predict(X_test))
print(knn.score(y_test=y_test))
```


<B><center>
## Methods
</center>
</B>

- ### <B><U>fit( X, y, n_neighbours=None, n_classes=None)</u></B>

>
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X {dtype:list/numpy_ndarray,shape:(n_samples,n_features)} | array like matrix denoting the input array  | 
>|    n_neighbours {dtype: int } | Number of neighbours for knn algorithm    |  
>|    n_classes{dtype:int}  | number of classes  for the classification |



- ### <B><U>predict( X)</u></B>

>
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X {dtype:list/numpy_ndarray,shape:(n_samples,n_features)} | array like matrix to predict.  | 


- ### <B><U> score(y_test)</u></B>

>returns the accuracy of the classification based on the predicted y_test values from the classifier
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    y_test {dtype:list/numpy_ndarray,shape:(n_samples)} | get metrics to evaluate our predictions from the model. | 


- ### <B><U>points()</u></B>

>returns the input X array given to the fit() function
>

- ### <B><U>labels()</u></B>

>returns the labels returned by the model after the model generates them.
>


You can view a sample implementation of KNN on ravml [here](https://github.com/ravenprotocol/ravml/blob/main/examples/knn_classifier.py)
