import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
import glob
from tensorflow.keras.models import load_model

IMAGE_SIZE = 68
TRAIN_DATASET = "dataset10/train/*"
TEST_DATASET = "dataset10/test1/*"


def proccess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = resize(image, IMAGE_SIZE)
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
    print(y_train)
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
    global fruit_labels

    MODEL = load_model("fruitDetectModel.h5")
    image = proccess_image(image)

    image = resize(image, IMAGE_SIZE)
    image = tf.keras.utils.normalize(image, axis=1)
    image = np.asarray([[image]])
    prediction = MODEL.predict(image)
    return (fruit_labels[np.argmax(prediction)], prediction)

def load_labels():
    train_filelist = glob.glob(TRAIN_DATASET)
    train_label_list = [name.split("_")[0].split("/")[-1] for name in train_filelist]
    return np.unique(train_label_list)


if __name__ == "__main__":
    global train_filelist, train_label_list
    fruit_labels = load_labels()

    train_filelist = glob.glob(TRAIN_DATASET)
    train_label_list = [name.split("_")[0].split("/")[-1] for name in train_filelist]
    fruit_labels = np.unique(train_label_list)
    x_train = import_images(train_filelist)
    x_train = np.asarray([proccess_image(image) for image in x_train])
    y_train = make_labels(train_label_list)

    test_filelist = glob.glob(TEST_DATASET)
    test_label_list = [name.split("_")[0].split("/")[-1] for name in test_filelist]
    x_test = import_images(test_filelist)
    x_test = np.asarray([proccess_image(image) for image in x_test])
    y_test = make_labels(test_label_list)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(220, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(180, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(60, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(len(fruit_labels), activation=tf.nn.softmax))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=15)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    predictions = model.predict([x_test])

    # for x in range(len(test_label_list)):
    #     pred_label = np.argmax(predictions[x])
    #     print("The label was:", pred_label, "and was", test_label_list[x])
    model.save("fruitDetectModel.h5")
else:
    fruit_labels = load_labels()
