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
import fruitDetect



TRAIN_DATASET = "dataset10/test1/*"
train_filelist = glob.glob(TRAIN_DATASET)
random.shuffle(train_filelist)
train_label_list = [name.split("_")[0].split("/")[-1] for name in train_filelist]
num_image = [cv2.imread(fname) for fname in train_filelist]

correct_ans = 0
count = 0
for x, image in enumerate(num_image):
    ans, pred = fruitDetect.detect_fruit(image)
    if train_label_list[x] == ans:
        correct_ans += 1
    count += 1

print(correct_ans / count )