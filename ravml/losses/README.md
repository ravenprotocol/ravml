<div align='center'><b>
<h1>

# Loss functions
</h1>
</div align></b>

### loss functions in ravml written using the ravop library compute the different loss values.

<div align='center'>

# mean_absolute_error

</div align>

Mean absolute error regression loss.

<div align='center'>
<img src=files/mean_abs_e.png width=250 height= 85>
</div align>

`Parameters:`<br>
><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.


`Returns:`<br>
>Ravop Tensor Object.

<br>


`Usage:`
 ```python
from ravml.losses import mean_absolute_error
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = mean_absolute_error(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```


<div align='center'>

# mean_squared_error
</div align>

    Mean squared error(MSE) measures the mean of the squared error 

<div align='center'>
<img src=files/meansquarederror.png width=250 height= 75>
</div align>
<u>

`Parameters:`<br></u>
><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.

`Returns:`<br></u>
>Ravop Tensor Object.

<br>    

`Usage:`

 ```python
from ravml.losses import mean_squared_error

# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = mean_squared_error(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```

<br><br>

<div align='center'>

# root_mean_squared_error
</div align>

    description of rmse
<div align='center'>
<img src=files/rootmeansquarederror.png width=250 height= 85>
</div align>



`Parameters:`<br>
><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.
    
`Returns:`<br>
>Ravop Tensor Object.

`Usage:`<br>

 ```python
from ravml.losses import root_mean_squared_error

# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = root_mean_squared_error(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```


<br><br>









<div align='center'>

# mean_squared_log_error
</div align>
    

<div align='center'>
<img src=files/rmsle.png width=250 height= 85>
</div align>


`Parameters:`<br>
><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.

`Returns:`<br>
>Ravop Tensor Object.

 ```python
from ravml.losses import mean_squared_log_error
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = mean_squared_log_error(y_true,y_pred)
''' returns a ravop tensor object ''' 

print('loss value:',loss_val())
 ```






<br><br>

<div align='center'>

# log_loss
</div align>

    Ravml implementation of the log loss function.

    
<div align='center'>
<img src=files/logloss.png width=280 height= 85>
</div align>
<br>

`Parameters:` <br>

><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)

`Returns:`<br></u>
>Ravop Tensor Object.

`Usage:`
 ```python
from ravml.losses import log_loss
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = log_loss(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```

<br><br>





<div align='center'>

# one_hot_cross_entropy
</div align>

    description of  one hot cross entropy

<div align='center'>
<img src=files/cross_entropy.png width=250 height= 45>
</div align>
<br>

`Parameters:`<br>
><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.


`Returns:`<br>
>Ravop Tensor Object.

`Usage:`
 ```python
from ravml.losses import one_hot_cross_entropy
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = one_hot_cross_entropy(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```

<br><br>






<div align='center'>

# sparse_cross_entropy
</div align>
    
    description of 

<div align='center'>
<img src=files/spe.png width=250 height= 85>
</div align>



`Parameters:`<br>
><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.

`Returns:`<br>
>Ravop Tensor Object.

`Usage:`
 ```python
from ravml.losses import sparse_cross_entropy
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = sparse_cross_entropy(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```

<br><br>



<div align='center'>

# categorical_hinge
</div align>


    calculates the categorical hinge loss between y_true and y_pred.

categorical hinge is formulated as :
<div align='center'>
<img src=files/hinge_loss.png width=350 height= 65>
</div align>


`Parameters:`<br>

><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.
    description of mse

`Returns:`<br>

>Ravop Tensor Object.


`Usage:`
<br>

 ```python
from ravml.losses import categorical_hinge
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = categorical_hinge(y_true,y_pred)
''' returns a ravop tensor object ''' 

print('loss value:',loss_val())
 ```


<br><br>

<div align='center'>

# Huber loss
</div>



Huber loss is formulated as:

<div align='center'>
<img src=files/huber_loss.png width=350 height= 85>
</div align>



`Parameters:`<br>



><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.

><i><b>d</b></i> : int value, d-value for the huber loss

`Returns:`<br>
>Ravop Tensor Object.

`Usage:`
 ```python
from ravml.losses import huber
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = huber(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```


<br><br>




<div align='center'>

# KL_div_loss
</div>
    
    The Kullback-Leibler divergence loss.
    

`Parameters:`<br>
><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.

`Returns:`<br>
>Ravop Tensor Object.

`Usage:`
 ```python
from ravml.losses import KL_div_loss
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = KL_div_loss(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```

<br><br>



<div align='center'>

# Poisson_loss
</div>



`Parameters:`<br>
><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.

`Returns:`<br>
>Ravop Tensor Object.

`Usage:`
 ```python
from ravml.losses import poisson_loss
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = poisson_loss(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```

<br><br>





<div align='center'>

# Logcosh 
</div>


`Parameters:`<br>
><b><i>y_true</i></b>: array-like of shape (n_samples,) or (n_samples, n_outputs)
Ground truth (correct) target values.

><i><b>y_pred</b></i> : array-like of shape (n_samples,) or (n_samples, n_outputs)
Estimated target values.
 
`Returns:`<br>
>Ravop Tensor Object.

<br><br>

`Usage:`
 ```python
from ravml.losses import logcosh
# y_true : actual y true labels.
# y_pred : prediction values from our model.

loss_val = logcosh(y_true,y_pred)
''' returns a ravop tensor object ''' 
 ```
