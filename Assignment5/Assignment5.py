import numpy as np
import sys
import os
import pandas as pd
from scipy import signal
from sklearn.metrics import accuracy_score



########## Note: Using Professor's code as a starting template  ########################


########################################
### Read images from train directory ###

traindir = sys.argv[1]

df = pd.read_csv(traindir+'/data.csv')#load images' names and labels
names = df['Name'].values
labels = df['Label'].values

traindata = np.empty((len(labels),3,3), dtype=np.float32)
for i in range(0, len(labels)):
	image_matrix = np.loadtxt(traindir+'/'+names[i])
#	traindata = np.append(traindata, np.array(image_matrix, ndmin=3, dtype=int8), axis=0)
#	traindata[i] = np.array(image_matrix, ndmin=3, dtype=np.int8)
	traindata[i] = image_matrix

print(traindata)
print(labels)


testdir = sys.argv[2]

df_test = pd.read_csv(testdir+'/data.csv')
test_name = df_test['Name'].values
test_label = df_test['Label'].values 

test_data = np.empty((len(test_label),3,3),dtype=np.float)

for i in range(0,len(test_label)):
    test_image_matrix = np.loadtxt(testdir+'/'+test_name[i])
    test_data[i] = test_image_matrix



sigmoid = lambda x: 1/(1+np.exp(-x))

##############################
### Initialize all weights ###

c = np.random.rand(2,2)
#c = np.ones((2,2), dtype=np.int8)

epochs = 1000
eta = .1
prevobj = np.inf
i=0

###########################
### Calculate objective ###
objective = 0



for i in range(0, len(labels)):
    hidden_layer = signal.convolve2d(traindata[i], c, mode="valid")
    for j in range(0, 2, 1):
        for k in range(0, 2, 1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    output_layer = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1])/4
    objective += (output_layer-labels[i]) ** 2
print('Output_layer = %s' % output_layer)



count = 0 
while(count < epochs):
    prevobj = objective
    dellc1 = 0
    dellc2 = 0
    dellc3 = 0
    dellc4 = 0
    f = output_layer**.5

    for i in range(0, len(labels)):
        hidden_layer = signal.convolve2d(traindata[i], c, mode="valid")
        for j in range(0, 2, 1):
            for k in range(0, 2, 1):
                hidden_layer[j][k] = sigmoid(hidden_layer[j][k])

        #gd c1
        sqrtf = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1])/4 - labels[i]
        dz1dc1 = hidden_layer[0][0] *(1 - hidden_layer[0][0]) *traindata[i][0][0]
        dz2dc1 = hidden_layer[0][1] *(1 - hidden_layer[0][1]) *traindata[i][0][1]
        dz3dc1 = hidden_layer[1][0] *(1 - hidden_layer[1][0]) *traindata[i][1][0]
        dz4dc1 = hidden_layer[1][1] *(1 - hidden_layer[1][1]) *traindata[i][1][1]
        dellc1 += (sqrtf * (dz1dc1 + dz2dc1 + dz3dc1 +dz4dc1))/2

        #gd c2
        dz1dc2 = hidden_layer[0][0] *(1 - hidden_layer[0][0]) *traindata[i][0][1]
        dz2dc2 = hidden_layer[0][1] *(1 - hidden_layer[0][1]) *traindata[i][0][2]
        dz3dc2 = hidden_layer[1][0] *(1 - hidden_layer[1][0]) *traindata[i][1][1]
        dz4dc2 = hidden_layer[1][1] *(1 - hidden_layer[1][1]) *traindata[i][1][2]
        dellc2 += (sqrtf * (dz1dc2 + dz2dc2 + dz3dc2 +dz4dc2))/2

        #gd c3
        dz1dc3 = hidden_layer[0][0] *(1 - hidden_layer[0][0]) *traindata[i][1][0]
        dz2dc3 = hidden_layer[0][1] *(1 - hidden_layer[0][1]) *traindata[i][1][1]
        dz3dc3 = hidden_layer[1][0] *(1 - hidden_layer[1][0]) *traindata[i][2][0]
        dz4dc3 = hidden_layer[1][1] *(1 - hidden_layer[1][1]) *traindata[i][2][1]
        dellc3 += (sqrtf * (dz1dc3 + dz2dc3 + dz3dc3 +dz4dc3))/2

        #gd c4
        dz1dc4 = hidden_layer[0][0] *(1 - hidden_layer[0][0]) *traindata[i][1][1]
        dz2dc4 = hidden_layer[0][1] *(1 - hidden_layer[0][1]) *traindata[i][1][2]
        dz3dc4 = hidden_layer[1][0] *(1 - hidden_layer[1][0]) *traindata[i][2][1]
        dz4dc4 = hidden_layer[1][1] *(1 - hidden_layer[1][1]) *traindata[i][2][2]
        dellc4 += (sqrtf * (dz1dc4 + dz2dc4 + dz3dc4 +dz4dc4))/2

    c[0][0] -= eta * dellc1
    c[0][1] -= eta * dellc2
    c[1][0] -= eta * dellc3
    c[1][1] -= eta * dellc4
    

    objective = 0
    for i in range(0, len(labels)):
        hidden_layer = signal.convolve2d(traindata[i], c, mode="valid")
        for j in range(0, 2, 1):
            for k in range(0, 2, 1):
                hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
        output_layer = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1]) / 4
        objective +=(output_layer - labels[i])**2
    
    print("i = %s  objective=%s" % (count, objective))
    count += 1


print("\n\nc=%s \n" %c)


# Final Pred Labels 
pred_labels = []
for i in range(0, len(test_label)):
    hidden_layer = signal.convolve2d(test_data[i], c, mode="valid")
    for j in range(0, 2, 1):
        for k in range(0, 2, 1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    output_layer = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1]) / 4
    print('Output_layer= %s' % output_layer)
    print('Test_data[i]= %s' % test_data[i])

    if (output_layer > 0.5):
        pred_labels.append(1)
    else:
        pred_labels.append(0)
print('\nTest Labels = %s' % test_label )
print('Predictions = %s' % pred_labels )
accuracy = accuracy_score(test_label, pred_labels) * 100
print('Accuracy = %s %%'% accuracy)


