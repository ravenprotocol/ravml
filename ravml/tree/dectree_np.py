import numpy as np

from sklearn.metrics import f1_score

def information_gain(X, y):
    info_gain = {}
    for i in range(len(X[0])):
        entropy_list = {}
        column = [k[i] for k in X]
        vals = (np.unique(column))
        for val in vals:
            temp = [y[row] for row in range(len(X)) if X[row][i] == val]
            entropy_list[val] = (entropy(temp))
        info_gain[i] = entropy(y) - np.sum(
            [column.count(i) / len(column) * entropy_list[i] for i in entropy_list.keys()])
    return info_gain


def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    class_prob = dict(zip(unique, counts))
    # print(class_prob)
    ent = [(i / len(y)) * np.log2(i / len(y)) for i in class_prob.values()]
    ent = -1 * np.sum(ent)
    return ent


class Node:
    def __init__(self, X, y, children=[]):
        self.X = X
        self.y = y
        self.children = [children]
        self.information_gain = None
        # self.entropy = None
        self.n = None
        self.depth=None
        self.label=None

    def set_leaf_node(self,label):
        self.label=label


class decision_tree(object):
    def __init__(self):
        self.max_depth = None
        self.max_split = None
        self.threshold = None
        self.root = None
        self.X = None
        self.y = None

    '''
    create the tree
    '''

    def _tree(self, X, y, cur_depth=0):
        self.node = Node(X, y)
        cur = self.root
        depth = 0
        if cur_depth <= self.max_depth:
            splits=self.get_split(X,y)
            for i in splits:
                self.node.children.append(self._tree(i[0],i[1],cur_depth+1))


        return cur

    '''
        check purity of the Node
    '''

    def get_split(self, X, y):
        splits=[]
        #split by criteria
        splits=[[X[:len(X)/2] ,y[:len(y)/2]], [X[len(X)/2:],y[:len(y)/2]] ]

        return splits

    def fit(self, X, y,max_depth):
        self.root= self._tree(X, y,max_depth)
    def predict(self, X):
        pass

    def traverse_tree(self):


class DecisionTreeClassifier(decision_tree):
    def __init__(self):
        super(DecisionTreeClassifier, self).__init__()

    def fit(self, X, y):
        dt = decision_tree()
        dt.fit(X, y)
        pass


data = [
    [0, 2, 1, 0],
    [0, 2, 1, 1],
    [1, 2, 1, 0],
    [2, 1, 1, 0],
    [2, 0, 0, 0],
    [2, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [2, 1, 0, 0],
    [0, 1, 0, 1],
    [1, 1, 1, 1],
    [1, 2, 0, 0],
    [2, 1, 1, 1]
]
y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
obj = DecisionTreeClassifier()
# print(information_gain(data, y))
print(obj.get_split(data, y))


def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini


from sklearn.tree import DecisionTreeClassifier as dt
