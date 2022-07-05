from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, \
    BatchNormalization, ReLU, AvgPool2D, Flatten, Dense, GlobalAvgPool2D

from tensorflow.keras import Model
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet import MobileNet


IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
N_BREEDS = 120

#  Globals
model = ''
info = ''


#  Function builds the model on a base of Keras included mobileNet, change the structure
#  of the model to be suitable for the model reviewed in the article.
#  this model will able to classify 120 classes
def build_model(train_batches, val_batches):
    global model
    mobile = tf.keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-6].output  # remove the 6 last layers of the keras mobilenet
    output = Dense(units=120, activation='softmax')(x)  # add dense layer with softmax activation function
    model = Model(inputs=mobile.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adamax(lr=0.0001), loss='categorical_crossentropy',
                  metrics=['accuracy'])  # compile the model
    model.fit(x=train_batches, steps_per_epoch=len(train_batches), validation_data=val_batches,
                        validation_steps=len(val_batches), epochs=15, verbose=1)  #  train the model
                                                                                    # with given data set batches


# Used as map function to each element of the data set, returns pairs of transforming image and its label
def preprocess(data):
    # Image conversion from int to float (change representation of pixels to be [0,1)
    img = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
    # resize image to fit in MobileNets model
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method='nearest')
    # encoding labels
    label = tf.one_hot(data['label'], N_BREEDS)

    return img, label


def prepare(data, batch_size):
    ds = data.map(preprocess)  # transform dataset into map uses preprocess as map_func
    ds = ds.shuffle(buffer_size=int(len(data)))  # randomly shuffles the elements in ds
    ds = ds.batch(batch_size)  # returns len(dataSet)/batch_size number of batches
    return ds


def prepare_image(file):  # making the image ready for processing
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def initiate():
    global model
    global info
    dataset, info = tfds.load(name="stanford_dogs", with_info=True)  # loading the data set "stanford_dogs"
    filename = 'trainedModel'  # we use this to name the trained data folder
    if os.path.isdir(filename):  # checks if we have trained data
        model = keras.models.load_model(filename)  # if we do we use the already trained model
    else:
        training_data = dataset['train']  # training data
        test_data = dataset['test']  # validation data
        train_batches = prepare(training_data, 32)
        val_batches = prepare(test_data, 32)
        build_model(train_batches, val_batches)  # train the model
        model.save(filename)  # saves the model for future use


def predict(path):
    global model
    global info
    get_name = info.features['label'].int2str  # int2str is a tensorflow function that convert integer into class name
    predictions = model.predict(prepare_image(path))  # predict wanted image with the model
    # Gets the class with the highest prediction rate
    top_component = tf.reshape(tf.math.top_k(predictions, k=1).indices, shape=[-1])
    top_matches = get_name(top_component[0])  # Gets the name of the predicted breed
    return top_matches

