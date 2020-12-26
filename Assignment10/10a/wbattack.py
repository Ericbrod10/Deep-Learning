from data import Data
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import sys
import tensorflow.keras as k
import numpy as np
import math


#Note Portion of Code from professor's Lecture and 
#https://www.tensorflow.org/tutorials/generative/adversarial_fgsm
#Which the professor said the clas could us

testDataFile = sys.argv[1]
testLabelsFile = sys.argv[2]
modelFile = sys.argv[3]
BatchSize = 1

dataObj = Data(BatchSize)

testData, testLabels = dataObj.loadTestData(testDataFile, testLabelsFile)

testData = testData /255

TargetModel = load_model(modelFile)

TargetModel.trainable = False


TensorLossObject = tf.keras.losses.CategoricalCrossentropy()

#Creating Adversarial structure code from tutorial
def create_adversarial_pattern(TargetModel, input_image, input_label):
    #input_image = preprocess(input_image)
    input_image = tf.cast(input_image, tf.float32)
    input_image = input_image/1.60
    with tf.GradientTape() as GradTape:
        GradTape.watch(input_image)
        prediction = TargetModel(input_image)
        loss = TensorLossObject(input_label, prediction)
    gradient = GradTape.gradient(loss, input_image)
    SignedGrad = tf.sign(gradient)
    return SignedGrad

#Setting Variables to keep track of model success and adv success

AdversarialPredCount = 0
ModelPredCount = 0

#Structure from example professor showed in class
for i in range(len(testData)):
    x = np.expand_dims(testData[i], axis=0)

    #p is normal model prediction
    p = TargetModel.predict(x)	
    pred = np.argmax(p)
    label = tf.reshape(testLabels[i], (1, TargetModel.predict(x).shape[-1]))
    gradient = create_adversarial_pattern(TargetModel, x, label)
    x_ = x +  gradient * 0.0625
    #x_ = x +  gradient * 0.0625
    #p_ is model prediction on adversial on image (x + 0.0625*gradient )
    p_ = TargetModel.predict(x_)	
    AdversarialPred = np.argmax(p_)
    true = np.argmax(testLabels[i])

    if true == AdversarialPred:
        print(i, '- \t',true, pred, AdversarialPred, '-  ', 'adv failed')
    else:
        print(i, '- \t',true, pred, AdversarialPred, '-  ', 'adv success')


    if pred == true:
        ModelPredCount += 1
    if AdversarialPred == true:
        AdversarialPredCount += 1

print('Accuracy of Target Model: %s%% ' % ((ModelPredCount / len(testData)) * 100))
print('Adversarial Accuracy: %s%%' % ((AdversarialPredCount /len(testData)) * 100))
