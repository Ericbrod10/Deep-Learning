import math
import tensorflow
import keras
import numpy as np
import os
import sys
from keras.models import load_model
from keras import backend, optimizers
from keras.utils import to_categorical

###Note some code from professor template and from class####
def generate_adversary(BlackboxModel, TrainData, TargetPreds):
    TrainData = TrainData / 4.5
    cost = backend.binary_crossentropy(TargetPreds, BlackboxModel.output)
    grad = backend.gradients(cost,BlackboxModel.input)
    calculategrads = backend.function(BlackboxModel.input, grad)
    gradients = calculategrads(TrainData)[0]
    adversaries = TrainData + epsilon * np.sign(gradients)
    return(adversaries)
learning_rate = 0.001
epsilon = 0.0625
split = 200
opt =  optimizers.Adam(learning_rate = learning_rate)



TestData = sys.argv[1]
TestLabels = sys.argv[2]
targetmodelfile = sys.argv[3]

BlackBox = sys.argv[4]
target_model = load_model(targetmodelfile)



TestData = np.load(TestData)
TestData = TestData.astype(np.float32) / (255.0)
TestLabels = np.load(TestLabels)

TestData = TestData[(TestLabels == 0) | (TestLabels == 1)]
TestLabels = TestLabels[(TestLabels == 0) | (TestLabels == 1)]
TestData = TestData - np.mean(TestData , axis = 0)

testdata = TestData
testlabels = to_categorical(TestLabels, num_classes = 2)

black_box_model = load_model(BlackBox)


target_model.compile(optimizer = opt, loss = 'binary_crossentropy',metrics = ['accuracy'])

TargetPreds = target_model.predict_classes(testdata)

adversaries = generate_adversary(black_box_model, testdata, to_categorical(TargetPreds, num_classes = 2))
    
TargetModelTestAccuracy = target_model.evaluate(testdata, testlabels)[1]
AdversarialTestAccuracy = target_model.evaluate(adversaries , testlabels)[1]

print("Target Model Accuracy: %s" % TargetModelTestAccuracy)
print("Adversary Test Accuracy: %s" % AdversarialTestAccuracy)
print("Accuracy Drop:", (TargetModelTestAccuracy - AdversarialTestAccuracy))
