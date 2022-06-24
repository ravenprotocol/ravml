

# Linear Regression

Linear Regression is a supervised machine learning technique with a continuous and constant slope projected output. Rather than aiming to classify data into categories (e.g. cat, dog), it is used to predict values within a continuous range (e.g. sales, price). There are two main types: Simple and Multivariable Regression.



<B><center>
## Methods
</center>
</B>

- ### <B><U>LinearRegression(x,y,theta)</u></B>

>initiates and fits the linear model for training data X,y.
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|    X {dtype:list/numpy_ndarray,shape:(n_samples,n_features)} | array like matrix denoting the input data |
>|    y {dtype:list/numpy_ndarray,shape:(n_samples)} | array like matrix denoting the input data | 


- ### <B><U>compute_cost()  </u></B>

>computes the cost function of the linear regression model



- ### <B><U>plot_graph(optimal_theta, res_file_path)  </u></B>

>plots the predictions by the linear regression model.
>
>| Parameters | Description     |
>| :------------ |:---------------:|
>|  optimal_theta {dtype:list/numpy_ndarray)} | array like matrix denoting the optimat values of the parameter theta. | 
>|    res_file_path {dtype:string} | filepath to save the plotted graph | 

<U>

## Example and results

</U>

```python
from ravml.linear import LinearRegression

iterations = 5
alpha = 0.01

algo = R.Graph(name='lin_reg', algorithm='linear_regression', approach='distributed')

model = LinearRegression(x,y,theta)
model.compute_cost()            
optimal_theta, inter, slope = model.gradient_descent(alpha, iterations)
print(optimal_theta, inter, slope)
res_file_path = str(pathlib.Path().resolve()) + '/result.png'
print(res_file_path)
model.plot_graph(optimal_theta, res_file_path)

algo.end()

```

<img src=lin_reg.png width=400>

You can view the implementation of Linear Regression [*here*](https://github.com/ravenprotocol/ravml/blob/main/ravml/linear/linear_regression.py).


