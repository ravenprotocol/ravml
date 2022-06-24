<center>

# <B> K-Nearest Neighbours </B>
</center>
Clustering is a type of unsupervised learning wherein data points are grouped into different sets based on their degree of similarity. The k-means clustering algorithm assigns data points to categories, or clusters, by finding the mean distance between data points. It then iterates through this technique in order to perform more accurate classifications over time. Since you must first start by classifying your data into k categories, it is essential that you understand your data well enough to do this

```python
#using kmeans for clustering using ravml
from ravml.cluster import KMeans

k = KMeans()
k.fit(X_train, 3, iter=30)
```

<!-- 
You can view a sample implementation of K-means on ravml [here](https://github.com/ravenprotocol/ravml/blob/main/examples/kmeans.py) -->

<B><center>
## Methods
</center>
</B>

- ### <B><U>fit(X,k=3,iter=100)</u></B>

>Compute kmeans clustering for thr input dataset X. where k denotes the number of centroids and iteration
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X {dtype:list/numpy_ndarray,shape:(n_samples,n_features)} | array like matrix denoting the input array for clustering | 
>|    k {dtype: int } | Number of centroids for clustering  |  
>|    iter{dtype:int ,default value: 10}  | number of iterations we want our model to update its centroids |  


- ### <U><B>predict(X)</B><br></U>

>   function to p redict the closest cluster each sample in X belongs to after centroids are updated.
>
>| Parameters | Description     |
>| :------------: |:---------------:|
>|    X {shape:(n_samples,n_features)} |New data to predict  | 


- ### <U> <B>score(X,y=None)</B><br></U>
>   Opposite of the value of X on the K-means objective.
>
>| Parameters | Description     |
>| :------------: |:---------------:|
>|    X {shape:(n_samples,n_features)} |New data to predict  | 
>|    y {shape:(n_samples),default:None} | input value of y_pred |

- ### <U> <B>set_params()</B><br></U>
>   get parameters related to the kmeans model.
>
>| Parameters | Description     |
>| :------------: |:---------------:|
>|    params  |New data to predict  | 

- ### <U><B>  plot()</B><br></U>
>plot the dataset after applying the ravml kmeans clustering algorithm .
>
>| Parameters | Description     |
>| :------------: |:---------------:|
>|    params  |New data to predict  | 

<BR>

<B><center>
## Example and Results
</center>
</B>



```python
from ravml.cluster import KMeans
from sklearn.model_selection import train_test_split
from dataset import load_data()

algo = R.Graph(name='kmeans', algorithm='kmeans',approach='distributed')

k = KMeans()
X = data
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
k.fit(X_train, 15, iter=20)
k.plot()

algo.end()
```

<!-- describe the code and results also deets-->
plot of the clusters formed after running the kmeans algorithm for 20 iterations :

<center>
<img src="kmeans.png"  width="400" title="new">
</center>

each different colour belongs to a different cluster. the plot shows the clusters formed after 10 iterations with k=15.

