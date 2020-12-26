import sys
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import normalize
from keras.models import *


#load in model
model = sys.argv[1]
gen = load_model(model)

#load noise parameters 
noise = np.random.normal(loc=0, scale=1, size=[100, 100])
genImages = gen.predict(noise)

genImages = genImages.reshape(100, 28, 28)


plt.figure(figsize=(10,10))


for i in range(genImages.shape[0]):
    plt.subplot(10, 10, i+1)

    plt.imshow(genImages[i], interpolation='nearest')
    plt.axis('off')

plt.tight_layout()

#load output save name as png and save the figure
output = sys.argv[2]
output = output + ".png"
plt.savefig(output)