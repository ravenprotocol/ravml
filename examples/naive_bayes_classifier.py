import sys


from naive_bayes import NaiveBayesClassifier

import random
from csv import reader

import numpy as np
import ravop.core as R



wine_data_url = "wine.data"

def load_csv(url):

    dataset = list()
    with open(url, 'r') as file:
        csv_data = reader(file)
        for row in csv_data:
            if not row:
                continue
            dataset.append(row)

    return dataset


df = load_csv(wine_data_url)
print("Dataset Downloaded")

# Convert string to float
for i in range(len(df)):
    for j in range(len(df[i])):
        df[i][j] = float(df[i][j])

def split_data(data, weight):
    
    """
    Randomly split data into train and tesst split
    """

    train_length = int(len(data) * (1 - weight))
    train = []
    for i in range(train_length):
        idx = random.randrange((len(data)))
        train.append(data[idx])
        data.pop(idx)
    return [train, data]

train, test = split_data(df, 0.2)

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(len(train)):
    y_train.append(train[i][0])
    X_train.append(train[i][1:])
    
for i in range(len(test)):
    y_test.append(test[i][0])
    X_test.append(test[i][1:])

model = NaiveBayesClassifier()



model.fit(X_train, y_train)
y_preds = model.predict(X_test)

calc_preds = []
for y_pred in y_preds:
    
    keys = list(y_pred.keys())
    # values = list(y_pred.values())
    #prediction = keys[np.argmax(np.asarray(values))]
    calc_pred = {key: y_pred[key]() for key in keys}
    calc_preds.append(calc_pred)

MAPs = []
for pred in calc_preds:
    
    MAP = max(pred, key= pred.get)
    MAPs.append(MAP)

print("NaiveBayesClassifier accuracy: {0:.3f}".format(model.accuracy(y_test, MAPs)))
