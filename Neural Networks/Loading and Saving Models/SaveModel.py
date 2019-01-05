import tensorflow as tf
from tensorflow import keras
import numpy as numpy
import numpy as np
from keras.models import load_model

fashionDataSet = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashionDataSet.load_data()

#convert pixels color to values between 1-0 (orginial 0-225)
train_images = train_images / 255.00
test_images = test_images / 255.00
print("Preparing pixel data for activation function")

print("Creating layers in Nuerel Network")
nuerelNetModel = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
	])

nuerelNetModel.compile(
	optimizer=keras.optimizers.Adam(),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
	)

epochs = input("Enter amount of epochs: ")
nuerelNetModel.fit(train_images, train_labels, epochs=int(epochs))

nuerelNetModel.save("my_model.h5")
print("Saved")
