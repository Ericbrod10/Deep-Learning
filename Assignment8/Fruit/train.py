import sys
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau)

#Batch_size, learn_rate, classes and epochs
batch_size = 32
learning_rate = 0.001
nb_classes = 101
nb_epoch = 30

#image rows and cols
img_channels = 3
img_rows = 32
img_cols = 32

#image shape
input_shape = Input((img_rows, img_cols, 3))

#Load in Train Dir
train_dir = sys.argv[1]

#Data Gen
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,featurewise_center=True)

#Training Data
train_generator = datagen.flow_from_directory(train_dir,subset='training',target_size=(img_rows, img_cols),batch_size=batch_size)

#Validation Data
validation_generator = datagen.flow_from_directory(train_dir,subset='validation',target_size=(img_rows, img_cols),batch_size=batch_size)

#Setting Model Checkpoint
mc = ModelCheckpoint('Best_Model', monitor='val_accuracy',mode='max',verbose=1, save_best_only=True)

model = applications.MobileNet(weights=None, input_tensor=input_shape, pooling='avg', classes=nb_classes)

#Optimize SGD
opt  = optimizers.SGD(lr=learning_rate , momentum=0.9)

#complile model
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


model.fit_generator(train_generator,epochs=nb_epoch,validation_data=validation_generator,callbacks=[mc],verbose=1)

model.save(sys.argv[2])