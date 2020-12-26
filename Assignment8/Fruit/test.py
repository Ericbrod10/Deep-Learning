import sys
import keras
import numpy as np
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


#loading in test dir
test_dir = sys.argv[1]

#img_row and cols
img_channels = 3
img_rows = 32
img_cols = 32

#DataGen
datagen = ImageDataGenerator(rescale=1./255,featurewise_center=True)

#Test Data
test_generator = datagen.flow_from_directory(test_dir,target_size=(img_rows, img_cols))

#load in model
model = load_model(sys.argv[2])

#complile Model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


#Finding Score
score = model.evaluate_generator(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', round(score[1], 4) * 100)