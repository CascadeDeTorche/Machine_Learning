
#import des differentes librairie
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from numpy.core.numeric import False_
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

#chemin vers les donnees

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

nb_image= len(os.listdir(allene_dir))+len(os.listdir(hex_dir))+len(os.listdir(cruciforme_dir))+len(os.listdir(plat_dir))
                                  

                                  
#commande pour afficher des images d'une certaines classe
#il faut remplacer 'allene/*' par le chemin vers les images qui nous interesse

# allene = list(data_dir.glob('allene/*'))
# PIL.Image.open(str(allene[0]))


#Chargement des images sur le disque en un dataset

batch_size = 25
img_height = 450
img_width  = 300

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.3),

  ]
)


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
           data_dir,                            #origine des fichiers
           validation_split=0.3,                #quel pourcentage des images est utilisÃ© en validation
           subset="training",                   #precise si c'est pour le training ou la validation
           seed=123,                            #seed pour le choix aleatoire des photos entre training et validation ?
           image_size=(img_height, img_width),  #dimension cible image
           batch_size=batch_size)               #ben la batch size

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
         data_dir,
         validation_split=0.3,
         subset="validation",                   #la on fait la validation
         seed=123,                              #faut la meme seed pour les deux je pense
         image_size=(img_height, img_width),
         batch_size=batch_size)


#Nom des classes d'apres le nom des dossiers
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

# definition d'un cache automatique pour les donnÃ©es, visiblement ca devrait limiter les problemes
# de saturation de memoire

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#normalisation des images

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))


model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 5, padding='same', activation='relu',input_shape=(img_height,img_width,3)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(4,activation='softmax')
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


#je conseil de lancer ca dans le terminal apres le script

#resume du model
#model.summary()

###entrainnement###

epochs=20
#history=model.fit(train_ds,epochs=epochs, validation_data=val_ds)

###Graphiques de precision###

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

###Enregistrement du modele###
# model.save("model_tool")

###Chargement d'un modele###
# model = keras.models.load_model("model_tool")




###Test du modele

# for i in range(5):
#     img_dir = "Test/Allen/ ("+str(i+1)+").jpg"
#     img_dir = pathlib.Path(img_dir)
    
#     img = keras.preprocessing.image.load_img(
#           img_dir,
#           target_size=(img_height, img_width))
    
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
    
#     predictions = model.predict(img_array)
#     score = max(predictions)
    
#     print(
#         "Cette vis allene a été estimée de la classe {} avec une confiance de {:.2f} %."
#         .format(class_names[np.argmax(score)], 100 * np.max(score)))
    
# for i in range(5):
#     img_dir = "Test/Hex/ ("+str(i+1)+").jpg"
#     img_dir = pathlib.Path(img_dir)
    
#     img = keras.preprocessing.image.load_img(
#           img_dir,
#           target_size=(img_height, img_width))
    
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
    
#     predictions = model.predict(img_array)
#     score = max(predictions)
    
#     print(
#         "Cette vis hexagonal a été estimée de la classe {} avec une confiance de {:.2f} %."
#         .format(class_names[np.argmax(score)], 100 * np.max(score)))
    
# for i in range(5):
#     img_dir = "Test/Cruciforme/ ("+str(i+1)+").jpg"
#     img_dir = pathlib.Path(img_dir)
    
#     img = keras.preprocessing.image.load_img(
#           img_dir,
#           target_size=(img_height, img_width))
    
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
    
#     predictions = model.predict(img_array)
#     score = max(predictions)
    
#     print(
#         "Cette vis cruciforme a été estimée de la classe {} avec une confiance de {:.2f} %."
#         .format(class_names[np.argmax(score)], 100 * np.max(score)))
    
# for i in range(5):
#     img_dir = "Test/Plat/ ("+str(i+1)+").jpg"
#     img_dir = pathlib.Path(img_dir)
    
#     img = keras.preprocessing.image.load_img(
#           img_dir,
#           target_size=(img_height, img_width))
    
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
    
#     predictions = model.predict(img_array)
#     score = max(predictions)
    
#     print(
#         "Cette vis plate a été estimée de la classe {} avec une confiance de {:.2f} %."
#         .format(class_names[np.argmax(score)], 100 * np.max(score)))
