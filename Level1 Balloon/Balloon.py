# -*- coding: utf-8 -*-
from sklearn import neighbors
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

f = open("adult+stretch.data","r")

#print (f.read())
data = f.readlines()

dataset = []

for line in data:
    dataset.append(line.split(","))

X = []
y = []

for case in dataset:
    new = case[0:4]
    if(new[0]=='PURPLE'): new[0]=1
    else: new[0]=0
    
    if(new[1]=='LARGE'): new[1]=1
    else: new[1]=0
    
    if(new[2]=='STRETCH'): new[2]=1
    else: new[2]=0
    
    if(new[3]=='ADULT'): new[3]=1
    else: new[3]=0
#    new[0],new[1],new[2] = 0,0,0
    print(new)
    X.append(new)    
    
    if(case[4] == 'T\n'): y.append('T')
    elif(case[4] == 'F\n'): y.append('F')
    else: y.append(case[4])
    
print (X)
print (y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)

my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

print("Accuracy: ", accuracy_score(y_test,predictions))