import keyboard
import cv2 
import sys
import time

image = []
fruit_type = None
counter = None
take_image = False


def handle_inputs():
    global take_image
    if keyboard.is_pressed("q"):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(-1)

    if keyboard.is_pressed(" "):
        print(take_image)
        if take_image: 
            take_image = False
        else: 
            take_image = True
        take_picture()
        time.sleep(0.1)

def take_picture():
    global counter
    name = f"{fruit_type}_{counter}"
    cv2.imwrite(f"preTest/{name}.jpg", image)
    print(name)
    counter += 1


def start(fruit_type_arg, counter_arg):
    global image
    global fruit_type
    global counter
    fruit_type = fruit_type_arg
    counter = int(counter_arg)

    cap = cv2.VideoCapture(0)
   

    while True:
        ret, frame = cap.read()
        image = frame
        cv2.imshow('frame',image)
        handle_inputs()
        ##if take_image: 
        ##    take_picture()
          ##  time.sleep(0.1)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image = []
    fruit_type = sys.argv[1]
    counter = sys.argv[2]
    start(fruit_type, counter)



    
