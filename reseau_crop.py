
#import des différentes librairie
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

#chemin vers les données

data_dir =     "Data_crop"
allene_dir =   "Data_crop/Allene"
hex_dir=       "Data_crop/Hex"
cruciforme_dir="Data_crop/Cruciforme"
plat_dir=      "Data_crop/Plat"

#nombre d'images disponibles
print('total image',              len(os.listdir(allene_dir))
                                  +len(os.listdir(hex_dir))
                                  +len(os.listdir(cruciforme_dir))
                                  +len(os.listdir(plat_dir)))
print('total images allene:',     len(os.listdir(allene_dir)))
print('total images hex:',        len(os.listdir(hex_dir)))
print('total images cruciforme:', len(os.listdir(cruciforme_dir)))
print('total images plat:',       len(os.listdir(plat_dir)))
print('\n')

#on transforme le chemin avec pathlib 
data_dir = pathlib.Path(data_dir)

#commande pour afficher des images d'une certaines classe
#il faut remplacer 'allene/*' par le chemin vers les images qui nous intéresse

# allene = list(data_dir.glob('allene/*'))
# PIL.Image.open(str(allene[0]))


#Chargement des images sur le disque en un dataset

batch_size = 32
img_height = 450
img_width  = 300

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
           data_dir,                            #origine des fichiers
           validation_split=0.3,                #quel pourcentage des images est utilisé en validation
           subset="training",                   #précise si c'est pour le training ou la validation
           seed=123,                            #seed pour le choix aléatoire des photos entre training et validation ?
           image_size=(img_height, img_width),  #dimension cible image
           batch_size=batch_size)               #ben la batch size

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
         data_dir,
         validation_split=0.3,
         subset="validation",                   #la on fait la validation
         seed=123,                              #faut la même seed pour les deux je pense
         image_size=(img_height, img_width),
         batch_size=batch_size)

#Nom des classes d'après le nom des dossiers
class_names = train_ds.class_names
print(class_names)


#pour visualiser des images de training et leurs labels

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# info sur les batches d'images (tailles, nombre d'image) 

# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

# définition d'un cache automatique pour les données, visiblement ça devrait limiter les problèmes
# de saturation de mémoire

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#normalisation des images

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu',input_shape=(img_height,img_width,3)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(4,activation='softmax')
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#je conseil de lancer ça dans le terminal après le script
#◘

#resumé du model
#model.summary()

#entrainnement
#model.fit(train_ds,epochs=5, validation_data=val_ds)

