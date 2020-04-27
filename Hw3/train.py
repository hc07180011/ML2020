import os
import sys
import cv2
import numpy as np
import tensorflow as tf

def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

workspace_dir = sys.argv[1]
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

from keras.utils.np_utils import *
train_y = to_categorical(train_y, 11)
val_y = to_categorical(val_y, 11)
train_x = train_x.astype(float) / 255.0
val_x = val_x.astype(float) / 255.0

'''
np.save(os.path.join(workspace_dir, 'train_x'), train_x.astype(float) / 255.0)
np.save(os.path.join(workspace_dir, 'train_y'), train_y)
np.save(os.path.join(workspace_dir, 'val_x'), val_x.astype(float) / 255.0)
np.save(os.path.join(workspace_dir, 'val_y'), val_y)
np.save(os.path.join(workspace_dir, 'test_x'), test_x.astype(float) / 255.0)
'''

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

np.random.seed(900104)
'''
data_path = sys.argv[1]
train_x = np.load(os.path.join(data_path, 'train_x.npy'))
train_y = np.load(os.path.join(data_path, 'train_y.npy'))
val_x = np.load(os.path.join(data_path, 'val_x.npy'))
val_y = np.load(os.path.join(data_path, 'val_y.npy'))
test_x = np.load(os.path.join(data_path, 'test_x.npy'))
'''

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1, l2
from keras.utils import multi_gpu_model
from keras import optimizers

batch_size = 128
epochs = 140

datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(train_x)
model = Sequential()
model.add(Conv2D(64, 3, input_shape=train_x[0].shape, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(11, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
                              steps_per_epoch=len(train_x)/batch_size,
                              epochs=epochs,
                              callbacks=callbacks_list,
                              validation_data=(val_x, val_y))

'''
model = load_model('best.h5')

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Accuracy.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Loss.png')
plt.close()

prediction = model.predict(test_x)
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, np.argmax(np.array(y))))
'''