# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
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

import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# Définition des répértoires d'images 
# Modifiez ces lignes pour indiquer le chemin d'accès aux données 
# à l'endroit où vous les avez enregistrées


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


# datagen = ImageDataGenerator(rotation_range=40,      # rotation de l'image originale
#                              width_shift_range=0.2,  # Translation horizontale et verticale de l'image
#                              height_shift_range=0.2,
#                              shear_range=0.2,        # Cisaillement aléatoire
#                              zoom_range=0.2,         # Zoom aléatoire
#                              horizontal_flip=True,   # Rotation à 90° de la moitié des données aléatoirement
#                              fill_mode='nearest')    # Stratégie à adopter pour le remplissage des pixels créés

# fnames = [os.path.join(train_cruciforme_dir, fname) for fname in os.listdir(train_cruciforme_dir)]


# img_path = fnames[3] # on choisit d'augmenter une image 
# img = image.load_img(img_path, target_size=(3264, 2448))



# représenttaion d'un exemple d'augmentation d'image

# x = image.img_to_array(img)
# x = x.reshape((1,) + x.shape)
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break          # le générateur tourne en boucle sur lui même, il faut indiquer une condition d'arrêt
# plt.show()








# # définition du modèle

# model = Sequential()

# # première couche de convolution
# model.add(Conv2D(32, (3, 3), activation='relu',
#           input_shape=(150, 150, 3)))

# #première couche de pooling
# model.add(MaxPooling2D((2, 2)))

# # deuxième couche de convolution
# model.add(Conv2D(64, (3, 3), activation='relu'))

# # deuxième couche de pooling
# model.add(MaxPooling2D((2, 2)))
# # ...
# model.add(Conv2D(128, (3, 3), activation='relu'))
# #...
# model.add(MaxPooling2D((2, 2)))
# #...
# model.add(Conv2D(128, (3, 3), activation='relu'))

# model.add(MaxPooling2D((2, 2)))

# # vectorisation de l'image résultante
# model.add(Flatten())

# model.add(Dense(512, activation='relu'))

# # Classification binaire 
# model.add(Dense(1, activation='sigmoid'))

# from keras import optimizers

# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])

# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale=1./255)
# validation_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(train_dir,               # On précise la localisation des images
#                                                     target_size=(150, 150),  # redimensionnement en 150*150
#                                                     batch_size=20,           # Batchs de 20 images 
#                                                     class_mode='binary')     # labels binaires 

# validation_generator = validation_datagen.flow_from_directory(validation_dir,
#                                                         target_size=(150, 150),
#                                                         batch_size=20,
#                                                         class_mode='binary')