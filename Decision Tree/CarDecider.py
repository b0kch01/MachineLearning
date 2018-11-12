# Not Really Useful
# First Machine Learning script tho! 

from sklearn import tree

print("Enter horse power")
HP_prediction = input()
print("Enter amount of seats")
S_prediction = input()

#[horsepower, Seasts]
specs = [[300, 2], [450, 2], [720, 2], [532, 2], [400, 4], [200, 8], [150, 9], [296, 7]]

#Corresponding Classes (Strings)
typeOfCar = ["sports-car", "sports-car", "sports-car", "sports-car", "sports-car", "minivan", "minivan", "minivan"]

#create a classifier
clf = tree.DecisionTreeClassifier()

#Fitness testing
clf = clf.fit(specs,typeOfCar)

#predictions
prediction = clf.predict([[HP_prediction, S_prediction]])
print("I believe that you have entered the specs of a: ", prediction)