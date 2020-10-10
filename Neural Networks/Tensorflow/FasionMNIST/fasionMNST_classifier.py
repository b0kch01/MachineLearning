# Comment from 9/14/2019 --> Wow. This is the first neural network program for classifying images.
# I was pretty clueless, but I won't touch the original comments just for history!

# My experience with the TensorFlow tutorial on "basic" classification. (i found this rly hard)
# Website: https://www.tensorflow.org/tutorials/keras/basic_classification
# Github: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb
# I realized I spelled a bunch of words wrong


# Importing Libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
# fixes a matlab bug
matplotlib.use('TkAgg')
from time import sleep
import matplotlib.pyplot as plt

# getting TensorFlow version and fetching dataset
print("TensorFlow version: " + tf .__version__ + "\n--------")
print("loading data...\n")
fashionDataSet = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashionDataSet.load_data()

# class names (y) 
classNames = ['t-shirt', 'pants', 'hoodie', 'dress', 'coat', 
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# Getting dataset details
print('(Number of images, pixel width, pixel height)')
print(train_images.shape)
print("Training Labels: " + str(len(train_labels)))
print()
print("(Number of images, pixel width, pixel height)")
print(test_images.shape)
print("Testing Labels: " + str(len(test_labels)))
print()

# asking user what to test
labrat = int(input("Test number: "))

# what it says below:
print("Seting up the first visual figure")
plt.figure()
plt.imshow(test_images[labrat])
plt.colorbar()
plt.grid(False)

# Making the classes in a visual format
print("Seting up the second visual figure")
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(classNames[train_labels[i]])

print("Would you like to see the visuals?")
answer = input("y/n: ")
if (answer == "y"):
	plt.show()

# convert pixels color to values between 1-0 (orginial 0-225)
train_images = train_images / 255.00
test_images = test_images / 255.00
print("Preparing pixel data for activation function")

print("Creating layers in Nuerel Network")
nuerelNetModel = keras.Sequential([
	# layer to convert images from 2D to 1D (not trainable)
	keras.layers.Flatten(input_shape=(28,28)),

	# Dense means that they are fully connected and trainable (the og layer)
	# units means the amount of nodes in the layer (circle things) 
	# activation is which activation (IDK what it is but relu is the best) Sigmoid, Tanh are also nice
	keras.layers.Dense(128, activation=tf.nn.relu),

	# this is the output layer where it creates an array of probabilites of the 10 items.
	keras.layers.Dense(10, activation=tf.nn.softmax)
	])

# compiling the nuerel net
nuerelNetModel.compile(
	# optimizer depends on the dataset/problem and adam is faster, but others work, but slower
	optimizer=tf.train.AdamOptimizer(),
	# loss function (sparse is used cuz the classes are not represented as vectors, but integers)
	loss='sparse_categorical_crossentropy',
	# there are types of units of measuring the preciseness of the model and accuracy is good enough.
	metrics=['accuracy']
	)

# asking user for epochs(you'll see later)
print("Done.")
epochs = input("Enter amount of epochs: ")
input("Press [Enter] to start training")
# making the presentation of the data nicer ;)
print("\n================================")
print("[TRAINING OUTPUT]")
print("================================\n")

# TRAINING TIME! so excited.
# ML peeps like to refer real life phyisical excercise terms to ML 
# loging the results (they are formatted in arrays) into a variable
#				  (x axis data,  y axis data,  amount of time to use whole data set)
epochsResult = nuerelNetModel.fit(train_images, train_labels, epochs=int(epochs))

# Let's grade the work!
# assigning these variables to the success rate of the nuerel net

print("\n================================")
print("[TESTING OUTPUT]")
print("================================\n")

testLoss, testAccuracy = nuerelNetModel.evaluate(test_images, test_labels)
print("Accuracy during testing: ", testAccuracy)

# get the previous history of the accuracy in each epoch and get the mean average
trainAccuracy = np.mean(epochsResult.history["acc"])
print("Accuracy during training: ", trainAccuracy)

# rip the testing accuracy is less than the training accuracy
# that means that we overfit our nuerel network. This can be fixed by using different
# functins and stuff but my knowledge is limited so maybe ill tackle this next time when i get more IQ

# LETS PREDICT!

print("\n================================")
print("[PREDICTION OUTPUT]")
print("================================\n")

# our model can now predict images! Lets use out testing images
predictions = nuerelNetModel.predict(test_images)
# we can get the output layer when we input a image (this case its the second test image)
print(predictions[labrat], "\n")
# we need the highest value becuase it's most likeley to be that
typeOfClothing = (np.argmax(predictions[labrat]))
# converts the integer into Enlish
print("The Nuerel Net says its a form of " + classNames[typeOfClothing] + ".")
print("It's a " + classNames[test_labels[labrat]])
