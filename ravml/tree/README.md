# Decision Tree 

Decison Tree are non parametric supervised learning methods used for classification and regression.


```python
from ravml.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()

#classifier fits on the train dataset
model.fit(X_train,y_train)

#predict on the test dataset
model.predict(X_test)

```


<U>

## Decision Tree Classifier
</U>

Decision Tree Classifier is capable of performing multi class classification on a dataset. The classification is performed using decision boundaries set by the the decision tree classifier.


>- ### <B><U>DecisionTreeClassifier(max_depth=5)</u></B>
>
>fit the decision tree classifier on the input training dataset X,y.
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X <br> <b>{dtype:int, default_value: 10} </b> | max depth of the decision tree| 



#  Methods



- ## <B><U>fit(X,y)</u></B>

### fit the decision tree classifier on the input training dataset X,y.
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X <br> <b>{dtype:list/numpy_ndarray , shape:(n_samples,n_features)} </b> | array like matrix denoting the input array for classification | 
>|    y <br> <b>{dtype:list/numpy_ndarray,shape:(n_samples)} </b>| classification output for the corresponding X values|
>|max_depth<br> <b>{dtype:int } </b> |<b><i>{default value:5}</i></b>|  
>|||


- ## <b>find_split()</b>
### find the best split of the decision tree so as to get maximum information gain.

>|Parameters|Description|
>|:-----|:---:|
>|X<br><i><b> {dtype:list/numpy_ndarray , shape:(n_samples in the  node,n_features in the node )} <b></i>|array like matrix representing the tree for which has to be split |
>|y<br><b><i>{shape:(n_samples,)}</b></i>|classification output for the corresponding X values |

- ## <b>grow_tree()</b>
### find the best split of the decision tree so as to get maximum information gain.

>|Parameters|Description|
>|:-----|:---:|
>|X<br><i><b> {dtype:list/numpy_ndarray , shape:(n_samples in the  node,n_features in the node )} <b></i>|tree for which has to be split |
>|y<br><b><i>{shape:(n_samples,)}</b></i>|classification output for the corresponding X values |


# Examples and usage

```python
from sklearn.datasets   load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


dataset = load_wine()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


obj = DecisionTreeClassifier(max_depth=3)

#fitting on training data.
obj.fit(X_train[:30], y_train[:30])

#predicting classification using predict function.
pr = obj.predict(X_test)

print(f1_score(y_test, pr, average='weighted'))

```


Decision tree implementation 