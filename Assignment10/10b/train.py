import math
import os
import sys
import keras
import numpy as np
import tensorflow
from keras.datasets.mnist import load_data
from keras import backend, losses, optimizers
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,  Dropout, Flatten, LeakyReLU
from keras.models import Model, Sequential, load_model


###Note some code is from template from class



TestData = sys.argv[1]
TestLabels = sys.argv[2]
TargetModel = sys.argv[3]
target_model = load_model(TargetModel)

blackboxmodel_save = sys.argv[4]

train_data = np.load(TestData)
train_data = train_data.astype(np.float32) / 255.0
train_labels = np.load(TestLabels)


train_data = train_data[(train_labels == 0) | (train_labels == 1)]
train_labels = train_labels[(train_labels == 0) | (train_labels == 1)]
train_data = train_data - np.mean(train_data, axis = 0)


learning_rate = 0.001
epsilon = 0.0625
opt =  optimizers.Adam(learning_rate = learning_rate)
split = 200
epochs = 10


data = train_data[:split,]
categorical_labels = to_categorical(train_labels[:split,], num_classes = 2)



#from template
loss_object = tf.keras.losses.CategoricalCrossentropy()
def generate_adversary(black_box_model, train_data, ModelPredictions):
    train_data = train_data / 2
    cost = backend.binary_crossentropy(ModelPredictions, black_box_model.output)
    grad = backend.gradients(cost, black_box_model.input)
    calculategrads = backend.function(black_box_model.input, grad)
    gradients = calculategrads(train_data)[0]
    adversaries = train_data + epsilon * np.sign(gradients)
    return(adversaries)





black_box_model = Sequential()
black_box_model.add(Conv2D(16, (3,3), activation='relu', input_shape=data.shape[1:]))
black_box_model.add(Flatten())
black_box_model.add(Dense(100, activation='relu'))
#black_box_model.add(Conv2D(16, (3,3), activation='relu', input_shape=data.shape[1:]))

#black_box_model.add(Conv2D(8, (3,3), activation='relu', input_shape=data.shape[1:]))

black_box_model.add(Dense(2, activation='sigmoid'))
black_box_model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])


target_model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
ModelPredictions = target_model.predict_classes(data)

target_categorical_predictions = to_categorical(ModelPredictions, num_classes = 2)
print("Target Model Accuracy: ", target_model.evaluate(data, categorical_labels)[1])

for epoch in range(epochs):

    black_box_model.fit(data, target_categorical_predictions, batch_size = 64, epochs = 10, shuffle = True, validation_split = 0.2, verbose = False)
    
    AdversaryData = generate_adversary(black_box_model, data, target_categorical_predictions)
    
    if AdversaryData.shape[0] < 6400:
        data = np.append(data , AdversaryData , axis = 0)
    else:
        data = AdversaryData

    data = data - np.mean(train_data , axis = 0)

    temp_target_prediction = data.shape[0] // categorical_labels.shape[0]
    ModelPredictions = np.tile(categorical_labels , (temp_target_prediction , 1))

    adversarial_predictions = target_model.predict_classes(data)
    target_categorical_predictions = to_categorical(adversarial_predictions, num_classes = 2)

    accuracy = target_model.evaluate(data, ModelPredictions)[1]
    print("Epoch:", epoch + 1, "Accuracy:", accuracy)

black_box_model.save(blackboxmodel_save)
