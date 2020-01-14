import csv
import pandas as pd
import numpy as np
from sklearn import svm


TRAIN_PROPORTION = 0.5

ref = pd.read_csv("outputTEST.csv")

train_index = int(len(ref["angry"])*TRAIN_PROPORTION)
print(train_index)

#c = np.concatenate(a, b)
X = np.column_stack((ref["angry"], ref["fear"], ref["happy"], ref["sad"], ref["surprise"], ref["neutral"], ref["disgust"]))
print(len(X))
print(len(ref["angry"]))
Y = ref["labels"]

X_neg = X[:172815]
X_pos = X[172816:]
Y_neg = Y[:172815]
Y_pos = Y[172816:]

X_neg_train = X_neg[:100000]
X_pos_train = X_pos[:100000]
Y_neg_train = Y_neg[:100000]
Y_pos_train = Y_pos[:100000]

X_neg_test = X_neg[100000:]
X_pos_test = X_pos[100000:]
Y_neg_test = Y_neg[100000:]
Y_pos_test = Y_pos[100000:]

X_train = np.concatenate((X_neg_train, X_pos_train))
X_test = np.concatenate((X_neg_test, X_pos_test))
Y_train = np.concatenate((Y_neg_train, Y_pos_train))
Y_test = np.concatenate((Y_neg_test, Y_pos_test))

clf = svm.SVC()
clf.fit(X_train, Y_train)

print(clf.predict([[1, 1, 1, 1, 1, 1, 1]]))

counter = 0
print(X[1865786:])
print(len(X))
neg_count = 0
pos_count = 0
true_count = 0
false_count = 0
for test_x in X_test:
    #print(clf.predict([test_x]))
    prediction = clf.predict([test_x])
    if prediction == "NEGATIVE":
        neg_count += 1
    if prediction == "POSITIVE":
        pos_count += 1
    if prediction == Y_test[counter]:
        true_count += 1
    if prediction != Y_test[counter]:
        false_count += 1

    counter += 1

print("OKOK")
print(neg_count)
print(pos_count)

print("TRUE AND AFALSEL")
print(true_count)
print(false_count)