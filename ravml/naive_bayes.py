import ravop.core as R
from ravop.core import Graph
from ravcom import inform_server
import time

class NaiveBayesClassifier(Graph):
    
    def __init__(self):
        pass

    def seperate_classes(self, X, y):

        """
        To seperate the dataset into suvbsets
        """

        seperated_classes = {}

        for i in range(len(X)):
            
            feature_values = X[i]
            class_name = y[i]

            if class_name not in seperated_classes:
                seperated_classes[class_name] = []
            seperated_classes[class_name].append(feature_values)

        return seperated_classes

    def stat_info(self, X):

        """
        Calculate mean and standard deviation
        """

        for feature in zip(*X):

            feature = R.Tensor(list(feature), name = 'feature')
            std = R.std(feature)
            mean = R.mean(feature)
            yield {
                'std': std,
                'mean': mean
            }

    def fit(self, X, y):
        
        """
        Train the model
        """

        seperated_classes = self.seperate_classes(X, y)
        self.class_summary = {}

        for class_name, feature_values in seperated_classes.items():

            self.class_summary[class_name] = {
                'prior_proba': len(feature_values)/len(X),
                'summary': [i for i in self.stat_info(feature_values)],
            }

        return self.class_summary

    def distribution(self, x, mean, std):

        """
        Get the Gaussian distribution
        """ 
        numerator = R.square(x - mean)
        denominator = R.Scalar(2) * R.square(std)
        frac = R.div(numerator,denominator)
        exponent = R.exp(R.Scalar(-1) * frac)
        two_pi = R.Scalar(2) *  Rpi()
        gaussian_denominator = R.square_root(two_pi) * std
        gaussian_func = R.div(exponent, gaussian_denominator)
        return gaussian_func

    def predict(self, X):

        """
        Predict the output class
        """

        MAPs = []

        for index, row in enumerate(X):
            joint_proba = {}

            for class_name, features in self.class_summary.items():
                total_features = len(features['summary'])
                likelihood = R.Scalar(1)

                for idx in range(total_features):
                    # print("feature: ", row[idx])
                    feature = R.Scalar(row[idx])
                    mean = features['summary'][idx]['mean']
                    stdev = features['summary'][idx]['std']
                    normal_proba = self.distribution(feature, mean, stdev)
                    likelihood = likelihood * normal_proba

                prior_proba = R.Scalar(features['prior_proba'])

                my_val = prior_proba * likelihood

                joint_proba[class_name] = prior_proba * likelihood
            MAPs.append(joint_proba)

        return MAPs

    
    def accuracy(self, y_test, y_pred):
        
        """
        Calculate model accuracy
        """

        true_true = 0

        for y_t, y_p in zip(y_test, y_pred):
            if(y_t == y_p):
                true_true += 1
        
        return true_true / len(y_test)