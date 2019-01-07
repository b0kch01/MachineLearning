from tensorflow import keras
import numpy as np
import matplotlib
#fixes a bug
matplotlib.use('TkAgg')
from time import sleep
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


fashionDataSet = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashionDataSet.load_data()

#class names (y) 
classNames = ['t-shirt', 'pants', 'hoodie', 'dress', 'coat', 
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

#asking user what to test
labrat = int(input("Test number: "))

#what it says below:
print("Seting up the first visual figure")
plt.figure()
plt.imshow(test_images[labrat])
plt.colorbar()
plt.grid(False)

#Making the classes in a visual format
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

#convert pixels color to values between 1-0 (orginial 0-225)
test_images = test_images / 255.00

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        nuerelNetModel = load_model('my_model.h5')

testLoss, testAccuracy = nuerelNetModel.evaluate(test_images, test_labels)
print("Accuracy during testing: ", testAccuracy)
predictions = nuerelNetModel.predict(test_images)
print(predictions[labrat], "\n")
print()
typeOfClothing = (np.argmax(predictions[labrat]))
print("The Neurel Net says its a form of " + classNames[typeOfClothing] + ".")
print("It's a " + classNames[test_labels[labrat]])
