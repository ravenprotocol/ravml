# Ravml Algorithms 

You can utilize the ravml algorithms in a graph by importing Graph from the ravop library .

```python
import ravop.core as R
from ravml import <algorithm>

#<algorithm> is the implementation of any available algorithm implemented in the ravml library

algo = R.Graph(name='svm', algorithm='svm', approach='distributed')

'''
implement your <algorithm> here

'''
algo.end()

```