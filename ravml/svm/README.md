# Support Vector Machine
SVM (Support Vector Machine) is a supervised machine learning technique that can be used to solve classification and regression problems. It is, however, mostly employed to solve categorization difficulties. Each data item is plotted as a point in n-dimensional space (where n is the number of features you have), with the value of each feature being the value of a certain coordinate in the SVM algorithm. Then we accomplish classification by locating the hyper-plane that clearly distinguishes the two classes.

```python
from ravml.svm import SVM_classifier
obj = SVM_classifier( learning_rate =0.01
         ,lamda_param=0.05
         ,gamma=0.2
         ,C=None
         ,coefficient=None
         ,degree=1
         ,n_iterations=20
         ,kernel='rbf')

```




>initiates and fits the linear model for training data X,y.
>
>| Parameters | Description     |
>| :------------- |:---------------:|
>|   learning_rate |<b>{default:0.01}</b>  |
>|  gamma  |  <b>{default:0.2}</b> parameter which decides the decision region.| 
>|n_iterations|<b>{default:10}</b> the number of iterations we want the algorithm to run. |
>|kernel|<b>{default:"rbf"}</b> parameter for selecting the kernel of the SVM function from (poly,rbf)|
>|C|<b>{default:1}</b> regularization parameter|

<U>

# Kernels
</U>



SVM uses Kernels for evaluating 

>| Kernels :|parameter value|
>|:--|:----:|
>|Radial basis function|'rbf'|
>|Polynomial kernel|'poly'|
>|Linear kernel|'linear'|
>|Sigmoid|'sigmoid'|


<U>

## Radial Basis Function kernel 
</U>
<p>

When training an SVM with the Radial Basis Function (RBF) kernel, two parameters must be considered:<b> C </b> and <b>gamma </b> . The parameter <b>C</b>, common to all SVM kernels, trades off misclassification of training examples against simplicity of the decision surface.
</p>

<center>
<img src=files/rbf.png width=200 height= 55>
</center>
<U>

## Polynomial kernel 
</U><p>

Polynomial kernel uses the <B>degree</B> parameter. the <b>degree</b> parameter determines the power of the polynomial function.
</p>
<u>

<center>
<img src=files/poly.png width=260 height= 50>
</center>


## Linear kernel 
</u><p>

this kernel uses the l2 norm distance to evalute the kernel function . It is most used when the features in the data can be seperated by a single line. It requires the SVM regularization parameter <B>C</B> for 
</p>
<center>
<img src=files/download.png width=240 height= 50>
</center>

<br><br>

<B><p font="times new roman">

# Methods

</B>

>- ### <B><U>fit(X,k=3,iter=100)</u></B>
>
>Compute kmeans clustering for thr input dataset X. where k denotes the number of centroids and iteration
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X <br> <b>{dtype:list/numpy_ndarray,shape:(n_samples,n_features)} </b> | array like matrix denoting the input array for classification | 
>|    k <br> <b>{dtype: int } </b>| Number of centroids for classification  |  
>|    iter<br> <b>{dtype:int ,default value: 10} </b> | number of iterations we want our model to update its centroids |  

<br>

>- ### <B><U>predict(X_test)</u></B>
>
>predicts the test dataset after the trained model is ready.
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X_test <br> <b>{dtype:list/numpy_ndarray,shape:(n_samples,n_features)}</b> | array like matrix denoting the input array for SVM prediction | 

<br>

>- ### <B><U>get_params()</u></B>
>return the parameters associated with the SVM model.
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>| N/A  | N/A |

<br>

>- ### <B><U>set_params()</U></B>
>
>function for setting the param
> <br>
>| Parameters | Description     |
>| :------------ |:---------------:|
>| N/A  | N/A |

SVC, NuSVC, SVR, NuSVR, LinearSVC, LinearSVR and OneClassSVM implement also weights for individual samples in the fit method through the sample_weight parameter. Similar to class_weight, this sets the parameter C for the i-th example to C * sample_weight[i], which will encourage the classifier to get these samples right. The figure below illustrates the effect of sample weighting on the decision boundary. The size of the circles is proportional to the sample weights:



# Example and implementation

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y =  datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33 ,random_state=42)
y = list(np.where(y == 0, -1, 1))

clf = SVM()
clf.fit(X, y)

clf.predict(X_test)
```
 