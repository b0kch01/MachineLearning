#required dependancies
import time
import os
from sklearn import tree
clear = clear = lambda: os.system('clear')

#Warning
print("Data set includes only adults.\nIf you are a kid, it will probably classify you as a female.\nPress [Enter] to continue")
input()
clear()

#Declaring Varibles for prediction later
print("Your hieght? (feet)")
height = input()
print("Your weight? (lb)")
weight = input()
print("Your shoe size? (US)")
shoeSize = input()


#[height, weight, shoeSize]
dataPoints = [[5.9, 176, 10.5], [5.8, 154, 9.5], [5.2, 132, 6], [5, 119, 5], [5.45, 143, 7.5], [6.2, 198, 13], [5.7, 141, 6.5],
     [5.8, 154, 7.5], [5.2, 121, 5], [5.6, 165, 8.5], [5.9, 187, 9.5]]

#Corresponding Classes (Strings)
gender = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#Creating Classifier and fitness training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(dataPoints, gender)

#Prediction
prediction = clf.predict([[height, weight, shoeSize]])
print("I believe you are a: ", prediction)