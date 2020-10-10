# Nathan Choi
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# Make a new "object" for standardscaling
sc = StandardScaler()
# Tune sc and then changing all the values
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Printing the amount of testing and training values
print("Train: {}\nTest: {}".format(len(y_train), len(y_test)))

# Function to design the model (using a function isn't neccessary but can look cleaner)
def create_model():
	# You know the usual
	nn = models.Sequential()
	nn.add(layers.Dense(32, activation="relu"))
	nn.add(layers.Dropout(0.5)) # This is to prevent overfitting
	nn.add(layers.Dense(16, activation="relu"))
	nn.add(layers.Dropout(0.5)) # So is this one
	nn.add(layers.Dense(8, activation="relu"))
	nn.add(layers.Dense(1, activation="sigmoid")) # Sigmoid becuase the activation returns a number 0-1
	return nn

model = create_model() # By running the createModel function, we can reference the model object as "model"\\

# Customizing the lost/cost function
model.compile(
	optimizer="adam", # Adam is just good
	loss="binary_crossentropy", # binary because its only 1 and 0
	metrics=["accuracy"] # accuracy works
)

# The model trains relitively fast so I made it quiet
history = model.fit(X_train, y_train, epochs=200, verbose=0) # Training the model. 50 is a good numbe1r of epochs
model.evaluate(X_test, y_test, verbose=0) # Testing it to test how it does in the real world

# Plot the data
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'acc'], loc='upper left')
plt.show()