import sys
import keras
import keras.backend as K
import numpy as np
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils, np_utils, to_categorical
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import (EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau)
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import AveragePooling2D, Conv2D, MaxPooling2D



img_rows = 256
img_cols = 256
batch_size = 32
learning_rate = 0.001
nb_epoch = 100

train_dir = sys.argv[1]
#train_dir = 'sub_imagenet/train/'
datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2,featurewise_center=True)

train_generator = datagen.flow_from_directory(train_dir,subset='training',target_size=(img_rows, img_cols),batch_size=batch_size)

validation_generator = datagen.flow_from_directory( train_dir,subset='validation',target_size=(img_rows, img_cols))


mc = ModelCheckpoint('Best_Model', monitor='val_accuracy', verbose=1, save_best_only=True)
input_shape = (img_rows, img_cols, 3)

base_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

layer = base_model.output
layer = Dropout(0.5)(layer)
layer = Flatten()(layer)
layer = Dense(256, activation='relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(128, activation='relu')(layer)
layer = Dropout(0.5)(layer) 
layer = Dense(128, activation='relu')(layer)
layer = Dropout(0.5)(layer)
output_layer = Dense(10, activation='softmax')(layer)

model = Model(base_model.input ,output_layer)

opt = keras.optimizers.Adagrad(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit_generator(train_generator,epochs=nb_epoch,callbacks=[mc],validation_data=validation_generator, validation_steps=batch_size,steps_per_epoch=batch_size,verbose=1)

model.save(sys.argv[2])