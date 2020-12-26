# Assignment 3 - Python

## Directions
Write a Python program that trains a single layer neural network
with sigmoid activation. You may use numpy. Your input is in dense 
liblinear format which means you exclude the dimension and include 0's. 

Let your program command line be:

python single_layer_nn.py <train> <test> <n>

where n is the number of nodes in the single hidden layer.

For this assignment you basically have to implement gradient
descent. Use the update equations we derived on our google document
shared with the class.

Test your program on the XOR dataset:

1 0 0
1 1 1
-1 0 1
-1 1 0

### Additional Assignment Questions
1. Does your network reach 0 training error? 
 - Yes. My my solution shows a zero error.

2. Can you make your program into stochastic gradient descent (SGD)?
 - It is possible to apply stochastic gradient descent in the program. Stochastic gradient descent is possible contingent on the use of by using subset of data to find the global minima. This solution could be faster than regular gradient descent since the process is applied on a smaller subset requiring less computing power.


3. Does SGD give lower test error than full gradient descent?
 - The error function of stochastic gradient descent is not minimized thus it will converge much faster relative to the error function of gradient descent and will give a greater test error.


4. What happens if change the activation to sign? Will the same algorithm
work? If not what will you change to make the algorithm converge to a local
minimum?
 - If you changed the activation to sign it pushes values toward -1 or +1. Instead, a ReLU could be used and will converge to a local minima.
