# coding: utf-8


import os
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
from sklearn import preprocessing
# jupyter path = "/home/jupyter/notebooks/python35"


BATCH_SIZE = 20
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
# NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
NUM_IMAGES = 98179
TRAINING_IMAGES_DIR = "/data1/tiny-imagenet-200/train/"
# TRAINING_IMAGES_DIR = './tiny-imagenet-200/train/'
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 9832
# VAL_IMAGES_DIR = './tiny-imagenet-200/val/'
VAL_IMAGES_DIR = "/data1/tiny-imagenet-200/val/"
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS




def load_training_images(image_dir, batch_size=500):
    image_index = 0

    images = np.ndarray(shape=(NUM_IMAGES, IMAGE_ARR_SIZE))
    names = []
    labels = []

    print("Loading training images from ", image_dir)
    # Loop through all the types directories
    for type in os.listdir(image_dir):
        if os.path.isdir(image_dir + type + '/images/'):
            type_images = os.listdir(image_dir + type + '/images/')
            # print(type_images)
            # Loop through all the images of a type directory
            batch_index = 0;
            # print ("Loading Class ", type)
            for image in type_images:
                image_file = os.path.join(image_dir, type + '/images/', image)
                print(image_file)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file)
                # print ('Loaded Image', image_file, image_data.shape)
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
        print ("path: ", testdir, image_file)

        # reading the images as they are; no normalization, no color editing
        image_data = mpimg.imread(image_file)
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



training_images, training_labels, training_files = load_training_images(TRAINING_IMAGES_DIR)

np.save('training_images', training_images)
np.save('training_labels', training_labels)
np.save('training_files', training_files)


# In[79]:


# training_images = np.load('training_images.npy')
# training_labels = np.load('training_labels.npy')
# training_files = np.load('training_files.npy')


# test
# load_validation_images(VAL_IMAGES_DIR, val_data)
os.listdir(VAL_IMAGES_DIR + '/images/')
os.path.join(VAL_IMAGES_DIR, 'images/')


shuffle_index = np.random.permutation(len(training_labels))
training_images = training_images[shuffle_index]
training_labels = training_labels[shuffle_index]
training_files = training_files[shuffle_index]

le_1 = preprocessing.LabelEncoder() #
training_le = le_1.fit(training_labels) #
training_labels_encoded = training_le.transform(training_labels)
print("First 30 Training Labels", training_labels_encoded[0:30])


val_data = pd.read_csv(VAL_IMAGES_DIR + 'val_annotations.txt', sep='\t', header=None,
                       names=['File', 'Class', 'X', 'Y', 'H', 'W'])


val_images, val_labels, val_files = load_validation_images(VAL_IMAGES_DIR, val_data, 1000)


np.save('val_images', val_images)
np.save('val_labels', val_labels)
np.save('val_files', val_files)


# val_images = np.load('val_images.npy')
# val_labels = np.load('val_labels.npy')
# val_files = np.load('val_files.npy')


#le_2 = preprocessing.LabelEncoder() #
#val_le = le_2.fit(training_labels) #
#val_labels_encoded = val_le.transform(val_labels)
val_labels_encoded = training_le.transform(val_labels)
print(val_labels_encoded[0:30])



height = IMAGE_SIZE
width = IMAGE_SIZE
channels = NUM_CHANNELS
n_inputs = height * width * channels
n_outputs = 200

reset_graph()



X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
y = tf.placeholder(tf.int32, shape=[None], name="y")

# input shape [-1, 64, 64, 3]
conv1 = tf.layers.conv2d(
    inputs=X_reshaped,
    filters=32,
    kernel_size=[5, 5],
    padding='SAME',
    activation=tf.nn.relu,
    name="conv1")

# shape after conv1: [-1, 64, 64, 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding='SAME',
    activation=tf.nn.relu,
    name="conv2")

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4)
dropout_reshape = tf.reshape(dropout, [-1, 8 * 8 * 64])

# Logits Layer
logits = tf.layers.dense(inputs=dropout_reshape, units=200, name='output')
Y_proba = tf.nn.softmax(logits, name="Y_proba")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

n_epochs = 50
batch_size = 20

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for batch in get_next_batch():
            X_batch, y_batch = batch[0], batch[1]
            # print ('Training set', X_batch.shape, y_batch.shape)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: val_images, y: val_labels_encoded})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./tiny_imagenet")


