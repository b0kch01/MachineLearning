#Uses grathwiz so make usre you get the require dependencies
#For Windows: https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi (make sure you add to PATH)
#For Mac: brew install graphviz

from sklearn import tree
import pydotplus
import os
import subprocess

#ask for input
print("Are you running on Windows?")
windows = input()
print("Enter horse power")
HP_prediction = input()
print("Enter amount of seats")
S_prediction = input()
print("Create dot file?")
createDot = input()

if createDot == "yes":
	print("Name of file?")
	name = input()

#[horsepower, seats]
specs = [[310, 2], [450, 2], [720, 2], [532, 2], [400, 4], [400, 4],
		 [200, 8], [150, 9], [296, 7], [260, 7], [287, 7], [280, 8], [375, 11], [153, 4]]

#Corresponding Classes (Strings)
typeOfCar = ["sports-car", "sports-car", "sports-car", "sports-car", "sports-car", "sports-car",
 			 "minivan", "minivan", "minivan", "minivan", "minivan", "minivan", "minivan", "minivan"]

#create a classifiers
classifier = tree.DecisionTreeClassifier()

#Fitness testing
classifier = classifier.fit(specs,typeOfCar)

#predictions
hundredIQ = classifier.predict([[HP_prediction, S_prediction]])
print("I believe that you have entered the specs of a: ", hundredIQ)

#classifier --> png (mac)
if (windows == "yes"):
	if createDot == "yes":
		print("Starting the creation of tree: " + name + ".dot")
		tree.export_graphviz(classifier,
										out_file=name + ".dot",
										feature_names=["Car horsepower", "Amount of seats"],
										class_names=["minivan", "sports-car"],
										filled = True,
										rounded = True)
		print("done")
		os.system("ls *.dot")

		#convert
		os.system("dot -Tpng " + name + ".dot -o " + name + ".png")
		print("converting to png...")
		os.system("ls *.png")
		print("done")

#classifier (windows)
else:
	if createDot == "yes":
		print("Starting the creation of tree: " + name + ".dot")
		tree.export_graphviz(classifier,
										out_file=name + ".dot",
										feature_names=["Car horsepower", "Amount of seats"],
										class_names=["minivan", "sports-car"],
										filled = True,
										rounded = True)
		print("done")
		os.system("dir  /s /b *.dot | findstr /e .dot")

		#converting
		os.system("dot -Tpng " + name + ".dot -o " + name + ".png")
		print("converting to png...")
		os.system("dir  /s /b *.png | findstr /e .png")
		print("done")