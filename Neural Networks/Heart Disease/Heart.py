# 5/7/2019 - this a an annotation on the kaggle Heart disease dataset
# https://www.kaggle.com/ronitf/heart-disease-uci

import os # Shhhhh tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import libraries
from tensorflow.keras import models, layers # import the modules for the model
import pandas as pd # To read the raw file
from sklearn.model_selection import train_test_split # To split the dataset in to training and testing
from sklearn.preprocessing import StandardScaler # It's a method to make all values between a certain range (0, 1)

# Get raw dataset and splitting X and y
dataset = pd.read_csv("heart.csv")
X = dataset.iloc[:, :-1].values # X is all collumns except the last one
y = dataset.iloc[:, -1].values # y is the last collumn

# Here is the basic syntax for splitting the x and y into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Make a new "object" for standardscaling
sc = StandardScaler()
# Tune sc and then changing all the values
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Printing the amount of testing and training values
print("Sizes:\nTrain: {}\nTest: {}".format(len(y_train), len(y_test)))

# Function to create the model (using a function isn't neccessary but can look cleaner)
def createModel():
	act = "relu"
	# You know the usual
	nn = models.Sequential()
	nn.add(layers.Dense(200, activation=act))
	nn.add(layers.Dropout(0.2)) # This is to prevent overfitting
	nn.add(layers.Dense(75, activation=act))
	nn.add(layers.Dropout(0.2)) # So is this one
	nn.add(layers.Dense(1, activation="sigmoid")) # Sigmoid becuase the model returns a number 0-1
	return nn


model = createModel() # Actually creating the model and representing it as "model"
model.compile(
	optimizer="adam", # Adam is just good
	loss="binary_crossentropy", # binary because its only 1 and 0
	metrics=["accuracy"] # accuracy works
	)

# The model trains relitively fast so I made it be quiet
model.fit(X_train, y_train, epochs=120, verbose=0) # Training the model. 120 is a good number of epochs
model.evaluate(X_test, y_test, verbose=0) # Testing it to get a more accurate accuracy
