# coding: utf-8

import os
import matplotlib
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.image as mpimg
from sklearn import preprocessing

BATCH_SIZE = 20
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
# NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
NUM_IMAGES = 98179
TRAINING_IMAGES_DIR = '../tiny-imagenet-200/train/'
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 9832
VAL_IMAGES_DIR = '../tiny-imagenet-200/val/'
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS


def get_directories():
    # datadir = os.environ["KRYLOV_DATA_DIR"]
    # principal = os.environ["KRYLOV_WF_PRINCIPAL"]

    # DATA_DIR = os.path.join(datadir, principal)
    # IMAGE_DIRECTORY = os.path.join(DATA_DIR, 'tiny-imagenet-200')
    TRAINING_IMAGES_DIR = '../tiny-imagenet-200/train/'  # os.path.join(IMAGE_DIRECTORY, 'train/')
    VAL_IMAGES_DIR = '../tiny-imagenet-200/val/'  # os.path.join(IMAGE_DIRECTORY, 'val/')

    return TRAINING_IMAGES_DIR, VAL_IMAGES_DIR


def load_training_images(image_dir, batch_size=500):
    image_index = 0

    images = np.ndarray(shape=(NUM_IMAGES, IMAGE_ARR_SIZE))
    names = []
    labels = []

    print("Loading training images from ", image_dir)
    # Loop through all the types directories
    for type in os.listdir(image_dir):
        if os.path.isdir(image_dir + type):
            type_images = os.listdir(image_dir + type)
            # Loop through all the images of a type directory
            batch_index = 0;
            # print ("Loading Class ", type)
            for image in type_images:
                image_file = os.path.join(image_dir, type, image)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file)
                print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :] = image_data.flatten()

                    labels.append(type)
                    names.append(image)

                    image_index += 1
                    batch_index += 1
                if (batch_index >= batch_size):
                    break;

    print("Loaded Training Images", image_index)
    return (images, np.asarray(labels), np.asarray(names))


def get_label_from_name(data, name):
    for idx, row in data.iterrows():
        if (row['File'] == name):
            return row['Class']

    return None


def load_validation_images(testdir, validation_data, batch_size=NUM_VAL_IMAGES):
    labels = []
    names = []
    image_index = 0

    images = np.ndarray(shape=(batch_size, IMAGE_ARR_SIZE))
    val_images = os.listdir(testdir + '/images/')

    # Loop through all the images of a val directory
    batch_index = 0;

    for image in val_images:
        image_file = os.path.join(testdir, 'images/', image)
        # print (testdir, image_file)

        # reading the images as they are; no normalization, no color editing
        image_data = mpimg.imread(image_file)
        print ('Loaded Image', image_file, image_data.shape)
        if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
            images[image_index, :] = image_data.flatten()
            image_index += 1
            labels.append(get_label_from_name(validation_data, image))
            names.append(image)
            batch_index += 1

        if (batch_index >= batch_size):
            break;

    print("Loaded Validation images ", image_index)
    return (images, np.asarray(labels), np.asarray(names))


def get_next_batch(batchsize=50):
    for cursor in range(0, len(training_images), batchsize):
        batch = []
        batch.append(training_images[cursor:cursor + batchsize])
        batch.append(training_labels_encoded[cursor:cursor + batchsize])
        yield batch


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# TRAINING_IMAGES_DIR, VAL_IMAGES_DIR = get_directories()
print(TRAINING_IMAGES_DIR, VAL_IMAGES_DIR)
training_images, training_labels, training_files = load_training_images(TRAINING_IMAGES_DIR)


shuffle_index = np.random.permutation(len(training_labels))
training_images = training_images[shuffle_index]
training_labels = training_labels[shuffle_index]
training_files = training_files[shuffle_index]

# np.save('training_images', training_images)
# np.save('training_labels', training_labels)
# np.save('training_files', training_files)

# training_images = np.load('training_images.npy')
# training_labels = np.load('training_labels.npy')
# training_files = np.load('training_files.npy')

le = preprocessing.LabelEncoder()
training_le = le.fit(training_labels)
training_labels_encoded = training_le.transform(training_labels)
print("First 30 Training Labels", training_labels_encoded[0:30])


# np.save('training_labels_encoded', training_labels_encoded)
# training_labels_encoded = np.load('training_labels_encoded.npy')

val_data = pd.read_csv(VAL_IMAGES_DIR + 'labels.txt', sep='\t', header=None,
                       names=['File', 'Class'])
val_images, val_labels, val_files = load_validation_images(VAL_IMAGES_DIR, val_data)
val_labels_encoded = training_le.transform(val_labels)
print(val_labels_encoded[0:30])

# np.save('val_images', val_images)
# np.save('val_labels', val_labels)
# np.save('val_files', val_files)
# val_images = np.load('val_images.npy')
# val_labels = np.load('val_labels.npy')
# val_files = np.load('val_files.npy')

# np.save('val_labels_encoded', val_labels_encoded)
# val_labels_encoded = np.load('val_labels_encoded.npy')

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 200
epochs = 20


x_train = training_images
y_train = training_labels_encoded
x_test = val_images
y_test = val_labels_encoded


x_train = x_train.reshape(98179, 12288)
x_test = x_test.reshape(9832, 12288)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train_ = keras.utils.to_categorical(y_train, num_classes)
y_test_ = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(12288,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train_,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test_))
score = model.evaluate(x_test, y_test_, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

