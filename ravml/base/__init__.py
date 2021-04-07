class Base(object):
    def __init__(self, **kwargs):
        self._params = {}
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self._params.update(**kwargs)

    @property
    def params(self):
        return self._params

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
