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
        X = R.Tensor(X)
        y = R.Tensor(y)
        num_classes = y.unique().shape_().gather(R.Scalar(0))
        num_features = X.shape_().gather(R.Scalar(1))
        while num_features.status != 'computed':
            pass
        self.num_classes = int(num_classes.output)
        self.num_features = int(num_features.output)
        print(self.num_classes, self.num_features)
        self.tree = self.grow_tree(X, y)

    def find_split(self, X, y):
        ideal_col = None
        ideal_threshold = R.Tensor([None])

        num_observations = y.shape_().gather(R.Scalar(0))
        while num_observations.status != 'computed':
            pass
        num_observations = int(num_observations.output)

        if num_observations <= 1:
            return ideal_col, ideal_threshold

        y = y.reshape(shape=R.Tensor([num_observations]))
        count_in_parent = R.Tensor([])
        for c in range(self.num_classes):
            count_in_parent = count_in_parent.concat(R.sum(R.equal(y, R.Scalar(c))).expand_dims())
        gini = R.square(count_in_parent.foreach(operation='div', params=num_observations))
        best_gini = R.sub(R.Scalar(1.0), R.sum(gini))
        temp_y = y.reshape(shape=R.Tensor([num_observations, 1]))

        for col in range(self.num_features):
            temp_X = R.gather(R.transpose(X), R.Scalar(col)).reshape(shape=R.Tensor([num_observations, 1]))
            all_data = R.concat(temp_X, temp_y, axis=1)

            column = R.gather(R.transpose(X), R.Scalar(col))
            ind = column.find_indices(R.sort(R.unique(column)))
            while ind.status != "computed":
                pass
            sorted_data = R.Tensor([])
            for i in ind.output:
                sorted_data = sorted_data.concat(all_data.gather(R.Tensor(i)))
            sorted_data_tpose = sorted_data.transpose()
            thresholds = sorted_data_tpose.gather(R.Scalar(0)).gather(R.Scalar(0))
            obs_classes = sorted_data_tpose.gather(R.Scalar(1)).gather(R.Scalar(0))

            num_left = R.Tensor([0] * self.num_classes)
            num_right = count_in_parent
            for i in range(1, num_observations):
                class_ = R.gather(obs_classes, R.Tensor([i - 1]))
                classencoding = R.one_hot_encoding(class_, depth=self.num_classes).gather(R.Scalar(0))
                num_left = num_left.add(classencoding)
                num_right = num_right.sub(classencoding)

                gini_left = R.sub(R.Scalar(1), R.sum(R.square(R.foreach(num_left, operation='div', params=i))))
                gini_right = R.sub(R.Scalar(1),
                                   R.sum(R.square(R.foreach(num_right, operation='div', params=num_observations - i))))
                gini = R.div(
                    R.add(R.multiply(R.Scalar(i), gini_left), R.multiply(R.Scalar(num_observations - i), gini_right)),
                    R.Scalar(num_observations))

                decision1 = R.equal(thresholds.gather(R.Tensor([i])), thresholds.gather(R.Tensor([i - 1])))
                decision2 = gini.less(best_gini)
                while decision2.status != "computed":
                    pass

                if decision1.output == 1:
                    continue

                if decision2.output == 1:
                    best_gini = gini
                    ideal_col = col
                    ideal_threshold = R.div(
                        R.add(thresholds.gather(R.Tensor([i])), thresholds.gather(R.Tensor([i - 1]))), R.Scalar(2))

        return ideal_col, ideal_threshold

    def grow_tree(self, X, y, depth=0):
        pop_per_class = R.Tensor([])
        for c in range(self.num_classes):
            pop_per_class = pop_per_class.concat(R.sum(R.equal(y, R.Scalar(c))).expand_dims())
        predicted_class = R.argmax(pop_per_class)
        while predicted_class.status != "computed":
            pass
        print(predicted_class,predicted_class.output)
        node = Node(predicted_class=predicted_class.output, depth=depth)
        node.samples = R.shape(y).gather(R.Scalar(0))
        if depth < self.max_depth:
            col=12
            threshold=R.Tensor([760])
            while threshold.status != "computed":
                pass
            z=X.shape_()
            z1=y.shape_()
            while z1.status!="computed":
                pass
            print(z,z1,X,y)
            if col is not None and threshold.output is not [None]:
                indices_left = X.transpose().gather(R.Scalar(col)).less(threshold)
                X_left = X.gather(R.find_indices(indices_left, R.Tensor([1])).reshape(shape=R.sum(indices_left).expand_dims() ) )
                y_left = y.gather(R.find_indices(indices_left, R.Tensor([1])).reshape(shape=R.sum(indices_left).expand_dims() ) )

                indices_right = X.transpose().gather(R.Scalar(col)).greater_equal(threshold)
                X_right = X.gather(R.find_indices(indices_right, R.Tensor([1])).reshape(shape=R.sum(indices_right).expand_dims() ))
                y_right = y.gather(R.find_indices(indices_right, R.Tensor([1])).reshape(shape=R.sum(indices_right).expand_dims() ))
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
                if obs[node.feature_index] < node.threshold.output:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.predicted_class)
        return predictions




