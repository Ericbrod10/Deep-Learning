# Assignment 8 - Python

## Directions
Classify images in the three Kaggle datasets on the course website 
with convolutional networks. You may use transfer learning. Your
goal is to achieve above 90% accuracy on the test/validation datasets.


Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the input training directory
and a model file name to save the model to. 

python train.py train <model file>

Make test.py take three inputs: the test directory 
and a model file name to load the model. 

python test.py test <model file>

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set.