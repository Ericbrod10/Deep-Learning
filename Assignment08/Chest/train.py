import os
from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
from keras import optimizers
import pandas as pd
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import sys


#Set Learn_rate, epochs and batch_size
learn_rate = 0.0001
epochs = 10
batch_size = 8


#load in train dir
train_dir = sys.argv[1]
img_rows  = 299
img_cols = 299


train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,validation_split=0.2,horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_dir,subset = 'training',target_size=(img_rows, img_cols),batch_size=batch_size,class_mode='categorical')

val_generator = train_datagen.flow_from_directory(train_dir,target_size=(img_rows,img_cols),subset = 'validation',batch_size = batch_size,class_mode = 'categorical')

res_model_base = InceptionResNetV2(weights='imagenet',include_top=False)

x = res_model_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=res_model_base.input, outputs=predictions)


#opt = keras.optimizers.Adadelta()
opt  = optimizers.SGD(lr=learn_rate, momentum=0.9)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


#Setting Model Checkpoint Checkpoint
mc = ModelCheckpoint('Best_Model', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model.fit_generator(train_generator,epochs=epochs,validation_data = val_generator,callbacks=[mc],shuffle=True)


model.save(sys.argv[2])

