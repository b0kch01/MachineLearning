#Took the iris data set and made a neural network to help predict the flower species based on its properties

#Import all the libraries, datatset, and frameworks
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn import datasets

#loading dataset
iris = datasets.load_iris()

#assigning data to variables (x axis, y axis), (input, output)
X = iris.data
Y = iris.target

#making an array with the clsasname so it is addressable as English later on
classNames = ['setosa', 'versicolor', 'virginica']

#Making the model and its layers
nuerelNetModel = keras.Sequential([
	#first layer has 128 nodes and uses the tanh activation function. I found out that it works better than other popular choices like sigmoid and relu
	keras.layers.Dense(128, activation=tf.nn.tanh),
  
	#output layer is a softmax so you can compare the different probabilities of "being that flower"
	keras.layers.Dense(3, activation=tf.nn.softmax)
	])

nuerelNetModel.compile(
	#for and optimizer to minimize loss, I use adam. Adam looks to have the most consistent results with high 90s in accuracy
	optimizer=tf.train.AdamOptimizer(),
  
	#loss function (sparse is used cuz the classes are not represented as vectors, but integers)
	loss='sparse_categorical_crossentropy',
  
	#there are types of units of measuring the preciseness of the model and accuracy is good
	metrics=['accuracy']
	)

#Making the neural net go through fitness training with the x and u dataset. I "feed" the data 20 times. Also stored the results in a variable 'epochsResult'
epochsResult = nuerelNetModel.fit(X, Y, epochs=20)

#caulculted accuracy with the same dataset because:
# 1. I don't know how to divide the dataset 
# 2. There is no seperate testing dataset
testLoss, testAccuracy = nuerelNetModel.evaluate(X, Y)

#forming a prediction and predicting
predictions = nuerelNetModel.predict(X)

print(predictions[149])
print(classNames[np.argmax(predictions[149])])
