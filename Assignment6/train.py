from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras import regularizers
import keras
import keras.backend as K
import numpy as np
import sys

batch_size = 32
nb_classes = 10
nb_epoch = 100

img_channels = 3
img_rows = 112
img_cols = 112

X_train = np.load(sys.argv[1])
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)




Y_train = np.load(sys.argv[2])
Y_train = to_categorical(Y_train, nb_classes)




print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)

model = Sequential()

#layer1
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', input_shape=[112,112,3], kernel_regularizer=regularizers.l2(0.0005)))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', kernel_regularizer=regularizers.l2(0.0005)))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.2))
#model.add(Conv2D(filters=64, kernel_size=(5,5)))
#model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))



#layer2
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', kernel_regularizer=regularizers.l2(0.0005)))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', kernel_regularizer=regularizers.l2(0.0005)))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.2))
#model.add(Conv2D(filters=32, kernel_size=(4,4)))
#model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))




#layer3
model.add(Conv2D(filters=112, kernel_size=(3,3), padding='valid', kernel_regularizer=regularizers.l2(0.0005)))
model.add(Activation('relu'))
model.add(Conv2D(filters=112, kernel_size=(3,3), padding='valid', kernel_regularizer=regularizers.l2(0.0005)))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.2))
#model.add(Conv2D(filters=64, kernel_size=(3,3)))
#model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))



#layer4
model.add(Conv2D(filters=112, kernel_size=(3,3), padding='valid', kernel_regularizer=regularizers.l2(0.0005)))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.2))
#model.add(Conv2D(filters=64, kernel_size=(2,2)))
#model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))


#layer5
model.add(Conv2D(filters=112, kernel_size=(3,3), padding='valid', kernel_regularizer=regularizers.l2(0.0005)))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.2))
#model.add(Conv2D(filters=64, kernel_size=(2,2)))
#model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))





#flatten layer
model.add(Flatten())
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))



#complile
model.compile(loss=keras.losses.categorical_crossentropy, 
optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#fit
model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            shuffle=True)


#save
model.save(sys.argv[3])