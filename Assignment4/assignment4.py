import numpy as np
import sys


# define sigmoid function to be used in later on in code:
def sigmoid(x): 
    return 1/(1+np.exp(-x))


#################
### Reads in data ###
# user passes in train data
f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:, 1:]
trainlabels = data[:, 0]

temp = np.ones((train.shape[0], 1))
train = np.append(train, temp, axis=1)

# user passes in test data
f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:, 1:]
testlabels = data[:, 0]

temp = np.ones((test.shape[0], 1))
test = np.append(test, temp, axis=1)

rows = train.shape[0]
cols = train.shape[1]

# User passes in batch size
mb_size = int(sys.argv[3])

# number of hidden nodes
hidden_nodes = 3

### Initialize all weights ###
w = np.random.rand(hidden_nodes)
W = np.random.rand(hidden_nodes, cols)

s = np.random.rand(hidden_nodes)
u = np.random.rand(hidden_nodes)
v = np.random.rand(hidden_nodes)

# Declare epochs and eta.
eta = 0.0001
epochs = 1000
stop = 0
prevobj = np.inf
i = 0

###########################
### Calculate objective ###
hidden_layer = np.matmul(train, np.transpose(W))


hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
output_layer = np.matmul(hidden_layer, np.transpose(w))

obj = np.sum(np.square(output_layer - trainlabels))

# Begin gradient descent

indices = np.array([i for i in range(rows)])

while(i < epochs):
# while(prevobj - obj > stop and i < epochs):
    prevobj = obj

    for k in range(0, rows, 1):
        # shuffle rows
        np.random.shuffle(indices)

        dellw = (np.dot(hidden_layer[indices[0], :], np.transpose(w)) - trainlabels[indices[0]]) * hidden_layer[indices[0], :]
        for j in range(1, mb_size):
            dellw += (np.dot(hidden_layer[indices[j], :], np.transpose(w)) - trainlabels[indices[j]]) * hidden_layer[indices[j], :]

	#update w
        w = w - (eta * dellw)


        # #dells
        dells = np.sum(np.dot(hidden_layer[indices[0], :], w) - trainlabels[indices[0]])*w[0]*(hidden_layer[indices[0], 0])*(1-hidden_layer[indices[0], 0])*train[indices[0]]
        for j in range(1, mb_size):
            dells += np.sum(np.dot(hidden_layer[indices[j], :], w) - trainlabels[indices[j]]) * w[0] * (hidden_layer[indices[j], 0]) * (1 - hidden_layer[indices[j], 0]) * train[indices[j]]
        
        
        #dellu
        dellu = np.sum(np.dot(hidden_layer[indices[0], :], w) - trainlabels[indices[0]])*w[1]*(hidden_layer[indices[0], 1])*(1-hidden_layer[indices[0], 1])*train[indices[0]]
        for j in range(1, mb_size):
            dellu += np.sum(np.dot(hidden_layer[indices[j], :], w) - trainlabels[indices[j]]) * w[1] * (hidden_layer[indices[j], 1]) * (1 - hidden_layer[indices[j], 1]) * train[indices[j]]
        

        #dellv
        dellv = np.sum(np.dot(hidden_layer[indices[0], :], w) - trainlabels[indices[0]])*w[2]*(hidden_layer[indices[0], 2])*(1-hidden_layer[indices[0], 2])*train[indices[0]]
        for j in range(1, mb_size):
            dellv += np.sum(np.dot(hidden_layer[indices[j], :], w) - trainlabels[indices[j]]) * w[2] * (hidden_layer[indices[j], 2]) * (1 - hidden_layer[indices[j], 2]) * train[indices[j]]


        dellW = np.empty((0, cols), float)
        dellW = np.vstack((dellW, dells, dellu, dellv))

	#update W
        W = W - (eta * dellW)

    # Recalculate objective
    hidden_layer = np.matmul(train, np.transpose(W))
    hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])


    output_layer = np.matmul(hidden_layer, np.transpose(w))
    obj = np.sum(np.square(output_layer - trainlabels))
    i = i + 1
    print('i = %s   Objective = %s' % (i, round(obj,5)))


### Print Final Predictions ###
finalpred = np.matmul(test, np.transpose(W))
predictions = np.sign(np.matmul(sigmoid(finalpred), np.transpose(w)))
error = (1 - (predictions == testlabels).mean())
error = round(error * 100, 5)
print('Predictions = %s' % predictions)
print('Error = %s' % error)
print('w = %s' % w)

