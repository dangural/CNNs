import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)

NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",  "dog", "frog", "horse", "ship", "truck"]

# show random images from train
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_train[random_index, :])
        ax.set_title(cifar10_classes[y_train[random_index, 0]])
plt.show()

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32,32,3)))
model.add(LeakyReLU(alpha=0.05))
BatchNormalization(axis=-1)

model.add(Conv2D(32, (3, 3)))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3)))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.1))

BatchNormalization(axis=-1)
model.add(Conv2D(64,(3, 3)))
model.add(LeakyReLU(alpha=0.05))

BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# Fully connected layer

BatchNormalization()
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.05))

BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(10))


model.add(Activation('softmax'))


model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test)
print()
print('Test accuracy: ', score[1])

predictions = model.predict_classes(x_test)

predictions = list(predictions)
actuals = list(y_test)

sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
sub.to_csv('./output_cnn2.csv', index=False)