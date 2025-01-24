import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


import pathlib
#data_dir = pathlib.Path('data')

#list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))

data_dir = pathlib.Path('data')

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))

image_count = len(list(list_ds))
print(image_count)

find = list(data_dir.glob('Leaf smut/*'))
PIL.Image.open(str(find[38]))



model = Sequential()

model.add(Conv2D(32, kernel_size = 3, input_shape = (64,64,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size = 3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size = 3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size = 3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(3))
model.add(Activation('softmax'))  # output layers


model.summary()
