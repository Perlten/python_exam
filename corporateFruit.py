import argparse
import cv2

import collect_data
import fruitDetect
import moveDetect
import fruitGraphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fruit detect')

    parser.add_argument("-g", '--graph', action='store_true', help='Show graph from csv file')
    parser.add_argument("-c", '--collect_data', nargs=2, help='Runs script to collect data')
    parser.add_argument("-i", "--image", nargs=1, help="Give direct path to image")
    args = parser.parse_args()

    if(args.graph):
        fruitGraphs.graph_file()
    elif(args.collect_data):
        collect_data.start(args.collect_data[0], args.collect_data[1])
    elif(args.image):
        image = cv2.imread(args.image[0])
        fruitDetect.detect_fruit(image)
    else:
        moveDetect.start()