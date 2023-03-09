import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


df = pd.read_csv('OnlineNewsPopularity/OnlineNewsPopularity.csv')
df = df.rename(columns=lambda x: x.strip())
df = df.iloc[: , 2:]

data = np.array(df)
x_data = data[:,:-1]
y_data = np.array([elem >= 1400 for elem in data[:,-1]])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=15)
clf = clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

print(f"sklearn accuracy: {accuracy}")