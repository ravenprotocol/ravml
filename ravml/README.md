# Ravml Algorithms 

You can implement the ravml algorithms in a graph by using importing Graph from ravop library .

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