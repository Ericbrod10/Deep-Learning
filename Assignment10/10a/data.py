import keras
import numpy as np


#Note Professor Sent us this script for reading in the data.

class Data:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.input_shape = (32, 32, 3)

    def loadTestData(self, testFile, testLabelsFile):
        testData = np.load(testFile)
        testLabels = np.load(testLabelsFile)

        testData = testData[np.logical_or(testLabels == 0, testLabels == 1)]
        testLabels = testLabels[np.logical_or(testLabels == 0, testLabels == 1)]
        testLabels = keras.utils.to_categorical(testLabels, 2)

        return testData, testLabels
