
import os

from keras.datasets import mnist
from matplotlib import pyplot

from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.python.util.nest import flatten

# Définition des répértoires d'images 
# Modifiez ces lignes pour indiquer le chemin d'accès aux données 
# à l'endroit où vous les avez enregistrées

tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# Chemin relatif, attention à vérifier que le dossier d'execution de python est le bon

train_dir =          os.path.abspath('Train')
train_allene_dir =   "Train/Allene"
train_hex_dir=       "Train/Hex"
train_cruciforme_dir="Train/Cruciforme"
train_plat_dir=      "Train/Plat"

validation_dir =          os.path.abspath('Validation')
validation_allene_dir =   os.path.abspath('Validation/Allene')
validation_hex_dir=       os.path.abspath('Validation/Hex')
validation_cruciforme_dir=os.path.abspath('Validation/Cruciforme')
validation_plat_dir=      os.path.abspath('Validation/Plat')


print('total training images allene:', len(os.listdir(train_allene_dir)))
print('total training images hex:', len(os.listdir(train_hex_dir)))
print('total training images cruciforme:', len(os.listdir(train_cruciforme_dir)))
print('total training images plat:', len(os.listdir(train_plat_dir)))
print('\n')
print('total validation images allene:', len(os.listdir(validation_allene_dir)))
print('total validation images hex:', len(os.listdir(validation_hex_dir)))
print('total validation images cruciforme:', len(os.listdir(validation_cruciforme_dir)))
print('total validation images plat:', len(os.listdir(validation_plat_dir)))


train_datagen = ImageDataGenerator(rescale=1/.255)
validation_datagen = ImageDataGenerator(rescale=1/.255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(816,612),class_mode="categorical")

validation_generator = train_datagen.flow_from_directory(validation_dir, target_size=(816,612),class_mode="categorical")

# Creation du model

model=Sequential()

#Première Convolution
model.add(Conv2D(32,(5,5),activation='relu',input_shape=(816,612,3)))
model.add(MaxPooling2D((2,2)))

#Deuxième Convolution
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(816,612,3)))
model.add(MaxPooling2D((2,2)))

#Vectorisation des images
model.add(Flatten())

#Reseau de neuronne 
model.add(Dense(1024,activation = 'relu'))
model.add(Dense(4,activation = 'softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_generator,epochs=5, validation_data=validation_generator)