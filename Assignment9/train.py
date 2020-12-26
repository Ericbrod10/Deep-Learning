import sys
import keras
import os
from PIL import Image  
from keras.datasets import mnist
import numpy as np
from keras.applications import *
from keras.utils import normalize
from keras.models import *
from keras.layers import *
from keras.initializers import *
from keras.optimizers import *

import matplotlib.pyplot as plt


####Note: Used Resources That Professor Gave the Class as a template#######


#Load in Mnist Data
(train, trainLabel), (test, testLabel) = mnist.load_data()
train = (train.astype(np.float32) - 127.5)/127.5


#Train Reshape
train = train.reshape(train.shape[0], train.shape[1] * train.shape[2])



#set epochs and batch_size
epochs = 450
batch_size=128


#image_size and latent_size
latent_size = 100
image_size = 784




#Set Discriminator
discriminator = Sequential()
#layer1
discriminator.add(Dense(1024, input_dim=image_size))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

#layer 2
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.2))

#Dense Layer
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dense(1, activation='sigmoid'))

#compliling discriminator
discriminator.compile(optimizer= Adam(lr=2e-4, beta_1=0.5), 
                        loss=keras.losses.binary_crossentropy, 
                        metrics=['accuracy']
                        )



#Set Generator
generator = Sequential()

generator.add(Dense(256, input_dim=latent_size))
generator.add(LeakyReLU(0.2))
#generator.add(LeakyReLU(0.1))
#generator.add(LeakyReLU(0.3))

##generator.add(Dense(256))
#generator.add(LeakyReLU(0.2))
#generator.add(LeakyReLU(0.1))
#generator.add(LeakyReLU(0.3))


generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
#generator.add(LeakyReLU(0.1))
#generator.add(LeakyReLU(0.3))


generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
#generator.add(LeakyReLU(0.1))
#generator.add(LeakyReLU(0.3))


#Added Dense Layer
generator.add(Dense(784, activation='tanh'))

#compiling generator
generator.compile(optimizer= Adam(lr=2e-4, beta_1=0.5), 
                    loss=keras.losses.binary_crossentropy, 
                    metrics=['accuracy']
                    )


discriminator.trainable=False
gan_input = Input(shape=(100,))

x = generator(gan_input)
gan_output= discriminator(x)

#Input and Put of Gan. Input from Generator -- Output from Discriminator
gan= Model(inputs=gan_input, 
            outputs=gan_output
            )

#Compiling Gan
gan.compile(loss=keras.losses.binary_crossentropy, 
            optimizer=Adam(lr=2e-4, beta_1=0.5)
            )

for e in range(1,epochs+1 ):
    for _ in range(batch_size):
        noise = np.random.normal(0,1, [batch_size, 100])
        GenImages = generator.predict(noise)
        ImageBatch = train[np.random.randint(low=0,high=train.shape[0],size=batch_size)]
        X = np.concatenate([ImageBatch, GenImages])
        
        YDis = np.zeros(2*batch_size)
        YDis[:batch_size] = 0.9
            
        discriminator.trainable = True
        discriminator.train_on_batch(X, YDis)
        #discriminator_loss = discriminator.train_on_batch(X, YDis)
        
        noise = np.random.normal(0,1, [batch_size, 100])
        YGen = np.ones(batch_size)
        
        discriminator.trainable = False
        
        gan.train_on_batch(noise, YGen)
        #gan_loss = gan.train_on_batch(noise, YGen)
    
    
    if e == 1 or e % 30 == 0:
        noise = np.random.normal(loc=0, scale=1, size=[100, 100])
        OutImages = generator.predict(noise)
        OutImages = OutImages.reshape(100,28,28)
        plt.figure(figsize=(10,10))
        for i in range(OutImages.shape[0]):
            plt.subplot(10, 10, i+1)
            plt.imshow(OutImages[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('GanTrainingImage%d.png' %e)
        #print(f'Epoch: {e} \t Discriminator Loss: {discriminator_loss} \t\t GAN Loss: {gan_loss}')
#Save Model
generator.save(sys.argv[1])