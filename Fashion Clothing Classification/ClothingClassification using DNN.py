

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist_dataset = tf.keras.datasets.fashion_mnist

""""

Calling load_data on this object will give two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items and their labels.

Total Images = 70000
60000- training
10000 - testing

"""

(training_images, training_labels), (test_images, test_labels) = mnist_dataset.load_data()



np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
plt.show()

print(training_labels[0])
print(training_images[0])


""""

All of the values in the number are between 0 and 255. 
If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1, 
Hence Normalizing the dataset

"""

training_images = training_images/255.0
test_images = test_images/255.0

# print(training_labels[0])
# print(training_images[0])



model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(128, activation=tf.nn.relu),
                            tf.keras.layers.Dense(10, activation = tf.nn.softmax)])


# - 512
# """""""""
# With 512
#
# Training Accuracy -
#
# [6.9529321e-08 1.3283780e-08 6.8448331e-09 2.1000401e-09 1.4136728e-09 4.5248237e-03 3.2638805e-08 1.3545395e-02 2.5544034e-08 9.8192960e-01]
# 9
# """"""


# model = tf.keras.Sequential([tf.keras.layers.Flatten(),
#                             tf.keras.layers.Dense(512, activation=tf.nn.relu),
#                             tf.keras.layers.Dense(10, activation = tf.nn.softmax)])


#- 1024

# With 1024
# Test Accuracy = 10000/10000 [==============================] - 0s 41us/sample - loss: 0.3440 - accuracy: 0.8743
# [1.1504925e-06 8.5605244e-08 4.1345191e-07 3.0787581e-09 1.0821088e-06 7.2051038e-04 6.5935046e-07 1.1445149e-02 3.1471664e-06 9.8782778e-01]
# 9


# by adding more Neurons we have to do more calculations,
# slowing down the process, but in this case they have a good impact -- we do get more accurate.
# That doesn't mean it's always a case of 'more is better', you can hit the law of diminishing returns very quickly!

# model = tf.keras.Sequential([tf.keras.layers.Flatten(),
#                             tf.keras.layers.Dense(512, activation=tf.nn.relu),
#                             tf.keras.layers.Dense(10, activation = tf.nn.softmax)])


#multiple Layers

# There isn't a significant impact -- because this is relatively simple data.
#
#
#
# Training time increases significantly
#
# 10000/10000 [==============================] - 1s 52us/sample - loss: 0.3320 - accuracy: 0.8802
# [2.8767585e-07 1.8291538e-06 2.0819039e-07 2.0412236e-07 1.5210333e-08 8.5554142e-03 2.6261620e-07 4.3631573e-03 2.0533960e-06 9.8707658e-01]
# 9

# Not much change in accuracy and probability of prediction

# model = tf.keras.Sequential([tf.keras.layers.Flatten(),
#                             tf.keras.layers.Dense(512, activation=tf.nn.relu),
#                             tf.keras.layers.Dense(256, activation=tf.nn.relu),
#                             tf.keras.layers.Dense(10, activation = tf.nn.softmax)])



"""""

Sequential: That defines a SEQUENCE of layers in the neural network

Flatten - Flatten just takes that square and turns it into a 1 dimensional set.

Dense: Adds a layer of neurons

Each layer of neurons need an activation function to tell them what to do.

Relu effectively means "If X>0 return X, else return 0",  it only passes values 0 or greater to the next layer in the network.

Softmax takes a set of values, and effectively picks the biggest one. 

if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05],
 
creates a one hot vector  [0,0,0,0,1,0,0,0,0]


This neural network has 3 layers:
Input Layer - which flattens the 2D representation of images into a 1 D array
Hidden Layer - has 128 neurons and activation function relu
Output Layer - has 10 neurons as there are 10 types of clothing labels so one neuron for each label and softmax activation function

"""



model.compile(optimizer=tf.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(training_images,training_labels,epochs=5)


# model.fit(training_images,training_labels,epochs=15)


#Evaluation of Model

model.evaluate(test_images,test_labels)



classifications = model.predict(test_images)

print(classifications[0])

print(test_labels[0])













"""""
15 epochs
10000/10000 [==============================] - 0s 32us/sample - loss: 0.3402 - accuracy: 0.8875
[9.1384287e-09 8.1735183e-12 2.7831111e-09 6.9799555e-10 1.0112613e-09 2.8234230e-05 1.6327718e-08 6.1651800e-02 2.6734498e-10 9.3831992e-01]
9
"""


"""


Results without normalization

Exploding gradient
Epoch 1/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.1994
Epoch 2/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0809
Epoch 3/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0521
Epoch 4/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0378
Epoch 5/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0280
313/313 [==============================] - 1s 2ms/step - loss: 0.0653
[1.5518777e-09 2.0957804e-10 2.0192276e-07 1.8620973e-05 6.2936845e-14 5.0040221e-09 9.0386293e-15 9.9998081e-01 1.6024381e-09 3.0965637e-07]
7
"""




# import tensorflow as tf
# print(tf.__version__)
#
# class myCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('accuracy')>0.6):
#       print("\nReached 60% accuracy so cancelling training!")
#       self.model.stop_training = True

#----------------------OR----------------------------------------

# class myCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('loss')<0.4):
#       print("\nReached 60% accuracy so cancelling training!")
#       self.model.stop_training = True
#
# callbacks = myCallback()
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# training_images=training_images/255.0
# test_images=test_images/255.0
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])