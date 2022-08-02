import ravop.core as R


# radial basis function
def rbf(X1, X2, gamma, **kwargs):
    l2_norm = R.norm(R.sub(X1, X2))
    d12 = R.pow(l2_norm, R.t(2))
    func = R.exp(d12.multiply(R.Scalar(-gamma)))
    return func()


# sigmoid kernel
def sigmoid(X1, X2, gamma, **kwargs):
    l2_dist = 0
    pass


#  function kernel
def polynomial_kernel(power, coef, **kwargs):
    def f(x1, x2):
        return (np.inner(x1, x2) + coef) ** power

    return f


def linear_kernel(**kwargs):
    pass


Kernels = {
    "rbf": rbf,
    "poly": polynomial_kernel,
    "linear": linear_kernel,
}
