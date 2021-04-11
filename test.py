from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import time


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def preprocess_images(data):
    data = data.reshape((-1, 28, 28, 1))
    data = data.astype('float32') / 255
    return data

train_images, test_images = preprocess_images(train_images), preprocess_images(test_images)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

t0 = time.time()
model.fit(train_images, train_labels, epochs=5, batch_size=32)
t1 = time.time()

print('Time taken to train: {}'.format(t1 - t0))
print(model.evaluate(test_images, test_labels))