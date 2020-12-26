import sys
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, BatchNormalization, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau)


#load batch_size, learn_rate, classes and epochs
batch_size = 8
learning_rate = 0.0001
nb_classes = 5
nb_epoch = 20

#img_rows and cols 
img_channels = 3
img_rows = 256
img_cols = 256

#setting input_shape
input_shape = Input((img_rows, img_cols, 3))


#loading in train dir
train_dir = sys.argv[1]

#dataGen
datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2 ,featurewise_center=True)

#train Data
train_generator = datagen.flow_from_directory(train_dir,subset = 'training',target_size=(img_rows, img_cols),batch_size=batch_size,class_mode='categorical')

#validation Data
validation_generator = datagen.flow_from_directory(train_dir,subset = 'validation',batch_size = batch_size,target_size=(img_rows, img_cols),class_mode='categorical')

#Creating model checkpoint
mc= ModelCheckpoint('Best_Model', monitor='val_accuracy',mode='max' , verbose=1, save_best_only=True)

transfer_model = applications.InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_shape)

layer = transfer_model.output
layer = GlobalAveragePooling2D()(layer)
layer = Dense(512, activation='relu')(layer)
predictions = Dense(nb_classes, activation='softmax')(layer)

model = Model(inputs=transfer_model.input, outputs=predictions)
opt  = optimizers.SGD(lr=learning_rate, momentum=0.9)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit_generator(train_generator,epochs=nb_epoch,validation_data=validation_generator,callbacks=[mc],verbose=1)

model.save(sys.argv[2])