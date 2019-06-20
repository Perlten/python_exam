import fruitDetect as fd
import glob
import numpy as np
import cv2


test_filelist = glob.glob("preTest/*")
print("glob")
test_label_list = [name.split("_")[0].split("/")[-1] for name in test_filelist]
print("labels made")
x_test = [cv2.imread(fname) for fname in test_filelist]

print("Images imported")

correct_ans = 0
for index, image in enumerate(x_test):
    print(f"{index} out of {len(x_test)}")
    label, prediction = fd.detect_fruit(image)
    print(f"Guess: {label} - Actually: {test_label_list[index]}")
    if label == test_label_list[index]:
        correct_ans += 1

print(f"Correct ans: {correct_ans} - % {correct_ans / len(x_test) }")

