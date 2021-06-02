from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from ravml.tree import DecisionTreeClassifier


dataset = load_wine()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

obj = DecisionTreeClassifier(max_depth=3)
obj.fit(X_train[:30], y_train[:30])
pr = obj.predict(X_test)

print(f1_score(y_test, pr, average='weighted'))