# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost"]
classifiers = [
    KNeighborsClassifier(1),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier()]

f = open("DataSet/leaf.csv","r")

#print (f.read())
data = f.readlines()

dataset = []

for line in data:
    dataset.append(line.split(","))

X = []
y = []

for case in dataset:
    new = case[2:14]
    X.append(new)    
    y.append(case[0])
    
#print (X)
#print (y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)

for i in range(len(names)):
    print(names[i])
    my_classifier = classifiers[i]
    my_classifier.fit(X_train, y_train)
    predictions = my_classifier.predict(X_test)
    print("Accuracy: ",accuracy_score(y_test,predictions))
