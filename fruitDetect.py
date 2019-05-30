import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import random
import pandas as pd
from matplotlib import pyplot as plt

IMAGE_SIZE = 50
TRAIN_DATASET = "dataset10/train/*"
TEST_DATASET = "dataset10/test1/*"
MODEL_NAME = "fruitDetectModel.h5"
BEST_MODEL_NAME = "fruitDetectModel_84P.h5"


def proccess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = tf.keras.utils.normalize([image], axis=1)[0]
    return image


def resize(image, new_dim):
    dim = (new_dim, new_dim)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

def make_labels(label_list):
    y_train = []
    for x in label_list:
        y_train.append(get_index_from_type(x))
    y_train = np.asarray(y_train)
    return y_train

def get_index_from_type(fruit_type:str):
    global fruit_labels

    for key, label in enumerate(fruit_labels):
        if label == fruit_type:
            return key


def import_images(filelist):
    num_image = np.array([resize(cv2.imread(fname), IMAGE_SIZE) for fname in filelist])
    return num_image

def detect_fruit(image):
    global fruit_labels, MODEL
    # return ("banana", [0,0,0,1,0,0,0])
    fruit_labels = load_labels()
    image = resize(image, IMAGE_SIZE)
    image = proccess_image(image)

    image = np.array(image).reshape(IMAGE_SIZE, IMAGE_SIZE, 1)

    # image = tf.keras.utils.normalize(image, axis=1)
    image = np.asarray([image])
    print(image.shape)
    # plt.imshow(image[0].reshape(IMAGE_SIZE,IMAGE_SIZE), cmap="gray")
    # plt.show()
    prediction = MODEL.predict(image)
    return (fruit_labels[np.argmax(prediction)], prediction)

def load_labels():
    global train_filelist, train_label_list
    train_filelist = glob.glob(TRAIN_DATASET)
    random.shuffle(train_filelist)
    train_label_list = [name.split("_")[0].split("/")[-1] for name in train_filelist]
    return np.unique(train_label_list)


def rotate_images(image_array, rotation):
    rotated_images = []
    for original_image in image_array:
        image = np.copy(original_image)
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rotation, 1)
        rotated90 = cv2.warpAffine(image, M, (h, w))
        rotated_images.append(rotated90)
    return np.asarray(rotated_images)


if __name__ == "__main__":
    global train_filelist, train_label_list
    fruit_labels = load_labels()

    x_train = import_images(train_filelist)
    x_train = np.asarray([proccess_image(image) for image in x_train])
    y_train = make_labels(train_label_list)

    # rotate_array1 = rotate_images(x_train, 90)
    # print(rotate_array1.shape)
    # rotate_array2 = rotate_images(x_train, 180)
    # print(rotate_array2.shape)
    # rotate_array3 = rotate_images(x_train, 270)
    # print(rotate_array3.shape)
    
    # print(len(y_train.shape))
    # x_train = np.concatenate((rotate_array1, rotate_array2, rotate_array3, x_train))
    # y_train = np.concatenate((y_train, y_train, y_train, y_train))
    # print(len(y_train))


    test_filelist = glob.glob(TEST_DATASET)
    test_label_list = [name.split("_")[0].split("/")[-1] for name in test_filelist]
    x_test = import_images(test_filelist)
    x_test = np.asarray([proccess_image(image) for image in x_test])
    y_test = make_labels(test_label_list)

    x_train = np.array(x_train).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y_train = pd.Series(y_train)
    y_train = pd.get_dummies(y_train.apply(pd.Series).stack()).sum(level=0)
    print(x_train[1])
    print(y_train[1])

    x_test = np.array(x_test).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y_test = pd.Series(y_test)
    y_test = pd.get_dummies(y_test.apply(pd.Series).stack()).sum(level=0)

    model = tf.keras.models.Sequential()
    model.add(Conv2D(64, (3,3), input_shape=x_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(len(fruit_labels)))
    model.add(Activation("sigmoid"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(x_train, y_train, batch_size=700, validation_split=0.1, epochs=3)
    val_loss, val_acc = model.evaluate(x_test, y_test)

    predictions = model.predict([x_test])
    print("pred:",predictions)
    for x in range(len(test_label_list)):
        pred_label = np.argmax(predictions[x])
        print("Thought it was:", fruit_labels[pred_label], "and was", test_label_list[x])
    model.save(MODEL_NAME)
else:
    MODEL = load_model(BEST_MODEL_NAME)