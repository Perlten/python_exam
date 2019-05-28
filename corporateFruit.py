import argparse
import cv2

import collect_data
import fruitDetect
import moveDetect
import fruitGraphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fruit detect')

    # Creates argument for graph
    parser.add_argument("-g", '--graph', action='store_true', help='Show graph from csv file')
    # Creates argument for starting our collect_data script
    parser.add_argument("-c", '--collect_data', nargs=2, help='Runs script to collect data')
    # Creates argument to scan a single image
    parser.add_argument("-i", "--image", nargs=1, help="Give direct path to image")
    args = parser.parse_args()

    if(args.graph):
        fruitGraphs.graph_file()
    elif(args.collect_data):
        collect_data.start(args.collect_data[0], args.collect_data[1])
    elif(args.image):
        image = cv2.imread(args.image[0])
        prediction = fruitDetect.detect_fruit(image)
        print(prediction)
    else:
        moveDetect.start()