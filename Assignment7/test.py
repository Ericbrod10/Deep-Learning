import keras
import keras.backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np
import sys


img_rows = 256
img_cols = 256

test_dir = sys.argv[1]
datagen = ImageDataGenerator(rescale=1./255,featurewise_center=True)

test_generator = datagen.flow_from_directory(test_dir,target_size=(img_rows, img_cols))

model = load_model(sys.argv[2])

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

score = model.evaluate_generator(test_generator)
print('Test loss:', round(score[0], 4))
print('Test accuracy:', round(score[1], 4) * 100)