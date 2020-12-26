# Assignment 10 - Python

## Directions
Implement a simple black box attack in Keras to attack a pretrained 
ResNet18 model from Keras. For the substitute model we use a two hidden 
layer neural network with each layer having 100 nodes.

Our goal is to generate adversaries to decieve a simple single layer 
neural network with 20 hidden nodes into misclassifying data from a 
test set that is provided by us. This test set consists of examples 
from classes 0 and 1 from CIFAR10. 

Your target model should have at least 85% accuracy on the test set without
adversaries. 

A successful attack should have a classification accuracy of at most 10%
on the test.

Submit your assignments as two files train.py and test.py. Make
train.py take three inputs: the test data, the target model to 
attack (in our case this is the network with 20 hidden nodes),
and a model file name to save the black box model file to.

python train.py <test set> <target model to be attacked> <black box model file> 

Your train.py program should output the accuracy of the target model on the
test data without adversaries as the first step. This is to verify that your
model has high accuracy on the test data without adversaries. Otherwise if your
model has low test accuracy it will be harder to attack.

When running train.py output the accuracy of the target model on the adversaries 
generated from the test data after each epoch.

Make test.py take three inputs: test set, target model, and the black box model.
The output should be the accuracy of adversarial examples generated with
epsilon=0.0625. A successful submission will have accuracy below 10%
on the advsersarial examples.

python test.py <test set> <target model to be attacked> <black box model file>
