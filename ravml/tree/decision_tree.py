import numpy as np
import ravop.core as R

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class Node:
    def __init__(self, predicted_class, depth=None):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.leftbranch = False
        self.rightbranch = False
        self.depth = depth


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.num_classes = None
        self.num_features = None

    def fit(self, X, y):
        self.num_classes = R.unique(R.Tensor(y)).shape_()
        self.num_features = R.Tensor(X).shape_()
        self.tree = self.grow_tree(X, y)

    def find_split(self, X, y):
        ideal_col = None
        ideal_threshold = None

        self.num_classes=np.unique(y).size
        num_observations = y.size

        if num_observations <= 1:
            return ideal_col, ideal_threshold

        y = y.reshape(num_observations, )

        count_in_parent = R.Tensor([np.count_nonzero(y == c) for c in range(self.num_classes)])




        gini = R.square(count_in_parent.foreach(operation='div', params=num_observations))
        best_gini = R.sub(R.Scalar(1.0) ,R.sum(gini))
        while best_gini.status!='computed':
            pass
        print(best_gini, "\n", count_in_parent)
        temp_y = y.reshape(y.shape[0], 1)

        for col in range(self.num_features):
            temp_X = X[:, col].reshape(num_observations, 1)
            all_data = np.concatenate((temp_X, temp_y), axis=1)
            sorted_data = all_data[np.argsort(all_data[:, 0])]
            thresholds, obs_classes = np.array_split(sorted_data, 2, axis=1)

            obs_classes = obs_classes.astype(int)

            num_left = [0] * self.num_classes
            num_right = count_in_parent.copy()

            for i in range(1, num_observations):
                class_ = obs_classes[i - 1][0]
                num_left[class_] += 1
                num_right[class_] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.num_classes))
                gini_right = 1.0 - sum((num_right[x] / (num_observations - i)) ** 2 for x in range(self.num_classes))
                gini = (i * gini_left + (num_observations - i) * gini_right) / num_observations
                if thresholds[i][0] == thresholds[i - 1][0]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    ideal_col = col
                    ideal_threshold = (thresholds[i][0] + thresholds[i - 1][0]) / 2
        return ideal_col, ideal_threshold

    def grow_tree(self, X, y, depth=0):
        pop_per_class = [np.count_nonzero(y == i) for i in range(self.num_classes)]
        predicted_class = R.argmax(pop_per_class)
        node = Node(predicted_class=predicted_class, depth=depth)
        node.samples = R.shape(y)

        if depth < self.max_depth:
            col, threshold = self.find_split(X, y)
            if col and threshold:
                indices_left = X[:, col] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                indices_right = X[:, col] >= threshold
                X_right, y_right = X[indices_right], y[indices_right]
                node.feature_index = col
                node.threshold = threshold
                node.left = self.grow_tree(X_left, y_left, depth + 1)
                node.left.leftbranch = True
                node.right = self.grow_tree(X_right, y_right, depth + 1)
                node.right.rightbranch = True
        return node

    def predict(self, X_test):
        node = self.tree
        predictions = []
        for obs in X_test:
            node = self.tree
            while node.left:
                if obs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.predicted_class)
        return np.array(predictions)


dataset = load_wine()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

obj = DecisionTreeClassifier(max_depth=7)
obj.find_split(X_train,y_train)
#obj.fit(X_train, y_train)
#pr = obj.predict(X_test)

#print(f1_score(y_test, pr, average='weighted'))
