# Assignment 7 - Python

## Directions
Write a convolutional network in Keras to train the Mini-ImageNet 
dataset on the course website. You may use transfer learning. Your
goal is to achieve above 90% accuracy on the test/validation datasets.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the input training directory
and a model file name to save the model to.

python train.py train <model file>

Make test.py take two inputs: the test directory
and a model file name to load the model.

python test.py test <model file>

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set.

In addition to your transfer learning solution also submit a
solution without transfer learning. In other words what is the
maximum test accuracy that you can obtain with a custom designed
model? Submit this as train2.py and test2.py with the same
parameters as above.
