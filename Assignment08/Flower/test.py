import sys
import keras
import numpy as np
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


#setting test dir
test_dir = sys.argv[1]

#img_rows and cols
img_channels = 3
img_rows = 256
img_cols = 256

#dataGen
datagen = ImageDataGenerator(rescale=1./255,featurewise_center=True)

#testGen
test_generator = datagen.flow_from_directory(test_dir,target_size=(img_rows, img_cols))

#loading in model
model = load_model(sys.argv[2])

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#Compute Score
score = model.evaluate_generator(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', round(score[1], 4) * 100)