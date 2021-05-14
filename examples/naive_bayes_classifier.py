from ravml.classifier.naive_bayes import NaiveBayesClassifier

import urllib2
import random
from csv import reader

wine_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

def load_csv(url):

    """
    Load the wine csv data
    """

    response = urllib2.urlopen(wine_data_url)
    csv_data = reader(response)

    dataset = []
    for row in csv_data:
        if not row:
            continue
        dataset.append(row)

    return dataset

df = load_csv(wine_data_url)

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

y_pred = model.predict(X_test)

print("NaiveBayesClassifier accuracy: {0:.3f}".format(model.accuracy(y_test, y_pred)))