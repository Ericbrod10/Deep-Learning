# Assignment 4 - Python

## Directions
Implement stochastic gradient descent (SGD) in your back propagation program that you wrote in assignment [3](./Assignment3). In the original SGD algorithm we update the gradient based on a single datapoint:

SGD algorithm:

Initialize random weights
for(k = 0 to n_epochs):
	Shuffle the rows (or row indices)
	for j = 0 to rows-1:
		Determine gradient using just the jth datapoint
		Update weights with gradient
	Recalculate objective

We will modify this into the mini-batch version and implement it for this
assignment.

I. Mini-batch SGD algorithm:

Initialize random weights
for(k = 0 to n_epochs):
	for j = 0 to rows-1:
		Shuffle the rows (or row indices)
		Select the first k datapoints where k is the mini-batch size
		Determine gradient using just the selected k datapoints
		Update weights with gradient
	Recalculate objective

Your input, output, and command line parameters are the same as assignment 3.
We take the batch size k as input. We leave the offset for the final layer 
to be zero at this time.

Test your program on the XOR dataset:

1 0 0
1 1 1
-1 0 1
-1 1 0

### Additional Assignment Questions
1. Test your program on breast cancer and ionosphere given on the website. Is the 
mini-batch faster or the original one? How about accuracy?


 - Batch-size has a high impact on the performance of mini-batch gradient descent, which can make it run slower or faster dependent on the size. Accuracy suffers relative to normal SGD since it may not be as stable. For the two particular datasets on the website, mini-batch gradient descent gave a better accuracy on breast cancer predictions if the data was standardized than ionosphere, but the normal non-standardized ionosphere data was more stable than non-standardized breast cancer  


2. Is the search faster or more accurate if you keep track of the best objective
in the inner loop?
 - The predictions would be more accurate and faster if we kept track of the best objective in the inner loop so we could cut it short if the objective started to get worse. As of right now the program could biased to a particular class and that could give us a wrong prediction. 

