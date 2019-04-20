import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def train_expert():
    @tf.function
    def train_step(image, label):
      with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(label, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
      train_loss(loss)
      train_accuracy(label, predictions)
      
    @tf.function
    def test_step(image, label):
      predictions = model(image)
      t_loss = loss_object(label, predictions)
    
      test_loss(t_loss)
      test_accuracy(label, predictions)
      
    EPOCHS = 5
    
    for epoch in range(EPOCHS):
      for image, label in train_ds:
        train_step(image, label)
    
      for test_image, test_label in test_ds:
        test_step(test_image, test_label)
    
      template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
      print (template.format(epoch+1,
                             train_loss.result(),
                             train_accuracy.result()*100,
                             test_loss.result(),
                             test_accuracy.result()*100))
def train_easy():
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs = 5)
    model.evaluate(x_test, y_test)

train_method = input("[ expert | beginner ]: ")
train_expert() if train_method == "expert" else train_easy()
