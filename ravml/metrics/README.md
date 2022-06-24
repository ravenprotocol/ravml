# Ravml metrics

Ravml has metrics function for evaluating our models in a distributed way. 


<div align='center'>

# accuracy
</div align='center'>
    accuracy is the ratio of the total number of predictions made correctly to the total number of samples predicted.

## Implementation:
```python
from ravml.metrics import accuracy
# y_true : actual y true labels.
# y_pred : prediction values from our model.
acc=accuracy(y_true,y_pred)
print("accuracy=",acc())
```


<div align='center'>

# r2_score
</div align='center'>

    r2_score(R-squared/coefficient of determination) is a metric used in case regression models to see how well the data fit the model.

## Implementation:
```python
from ravml.metrics import r2_score
# y_true : actual y true labels.
# y_pred : prediction values from our model.
r2=r2_score(y_true,y_pred)
print("r2_score=",r2())
```
<div align='center'>

# get_TP_TN_FN_FP
</div align='center'>
    get_TP_TN_FN_FP() function returns the metrics True positive , True Negative ,False Negative and False Positive in a list.


## Implementation:
```python
from ravml.metrics import get_TP_TN_FN_FP
# y_true : actual y true labels.
# y_pred : prediction values from our model.
metric=get_TP_TN_FN_FP()
print(" True Positive",metric[0]())
print(" True Negative",metric[1]())
print(" False Negative",metric[2]())
print(" False Positive",metric[3]())
```
<div align='center'>

# precision
</div align='center'>
    
    Precision returns the proportions of the positive predictions that were actually correct.

    Precision= True Positives / (True Positives + False Positives)

## Implementation:

```python
from ravml.metrics import precision
# y_true : actual y true labels.
# y_pred : prediction values from our model.
metric=precision()
print("evaluated metric",metric())
```
<div align='center'>

# recall
</div align='center'> 
    
    Precision returns the proportions of the actual positives that were identified correctly.

    Precision= True Positives / (True Positives + False Negatives)

## Implementation:

```python
from ravml.metrics import recall
# y_true : actual y true labels.
# y_pred : prediction values from our model.
metric=recall()
print("evaluated metric",metric())
```
<div align='center'>

# f1_score
</div align='center'>

The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean. It is primarily used to compare the performance of two classifiers.


## Implementation:

```python
from ravml.metrics import f1_score
# y_true : actual y true labels.
# y_pred : prediction values from our model.
metric=f1_score()
print("evaluated metric",metric())
```