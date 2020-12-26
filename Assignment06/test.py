from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
import keras
import keras.backend as K
import numpy as np
import sys

nb_classes = 10
X_test = np.load(sys.argv[1])
Y_test = np.load(sys.argv[2])
Y_test = to_categorical(Y_test, nb_classes)

img_channels = 3
img_rows = 112
img_cols = 112

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)


model = load_model(sys.argv[3])

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', round(score[0], 4))
print('Test accuracy:', round(score[1], 4) * 100)
