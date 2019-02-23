# Import all the libraries, datatset, and frameworks
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random

# Getting the prepared dataset from sklearn
iris = datasets.load_iris()

# assigning data to variables (x axis, y axis), (input, output)
X = iris.data
y = iris.target

# Splitting the dataest into two sections: Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Making an array with the classnames so it is addressable as English later on
#			 [	  0   ,       1     ,       2    ]
classNames = ['Setosa', 'Versicolor', 'Virginica']

# Creating the model and its layers
neural_net_model = keras.Sequential([
    # First layer has 30 nodes and uses the tanh activation function. I found out that it works better than sigmoid and relu
    # 30 Nodes feel like a bit crazy or this dataset, but it gives promising results
    keras.layers.Dense(30, activation=tf.nn.tanh),
    # output layer is a softmax so you can compare the different probabilities of "being that flower"
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

neural_net_model.compile(
    # For an optimizer to minimize loss, I use adam. Adam looks to have the most consistent results
    optimizer=tf.train.AdamOptimizer(),

    # Loss function (this is used because of how the dataset is formatted)
    loss='sparse_categorical_crossentropy',

    # For classification, accuracy is the most "accurate" unit to measure accuracy
    metrics=['accuracy']
)

# Making the neural net go through fitness training with the x and y dataset. I "feed" the data 40 times.
# For small datset, you need alot more epochs
training_results = neural_net_model.fit(X_train, y_train, epochs=40, verbose=0)
print("Training Accuracy:",
      str(round(training_results.history["acc"][-1] * 100, 2)) + "%")

# Calculted accuracy with the test data:
testLoss, testAccuracy = neural_net_model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy:", (str(round(testAccuracy * 100, 2)) + "%"))

# Testing the Model with our predictions
predict = neural_net_model.predict(X)
random_number = random.randint(0, 150)

print("Testing for", classNames[y[random_number]])
print("Computer is " + str(round(np.amax(predict[random_number]) * 100)) +
      "% sure that it is " + classNames[np.argmax(predict[random_number])])
