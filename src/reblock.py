'''
- usage: python reblock.py [H] [W] [D]
- iterate through image directory and regenerate the blocks in blocks/
- will use blockify method in utils
'''

import os

# import the necessary packages
import numpy as np
import argparse
import cv2
from utils import blockify

ap = argparse.ArgumentParser()
ap.add_argument("-H", "--rows", required=True,
    help="num rows")
ap.add_argument("-W", "--columns", required=True,
    help="num columns")
ap.add_argument("-D", "--depth", required=True,
    help="num channels")

args = vars(ap.parse_args())

block_rows = int(args["rows"])
block_cols = int(args["columns"])
block_depth = int(args["depth"])

if __name__ == "__main__":

    if not os.path.exists("data/blocks"):
        os.makedirs("data/blocks")

    if not os.path.exists("data/images"):
        os.makedirs("data/images")
        print("created images directory, please put in images")

    for filename in os.listdir("data/images"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread("data/images/" + filename, 4)
            if (len(image.shape) == 2):
                image = np.expand_dims(image, axis = 2)
            blocks = blockify(image, (block_rows, block_cols, block_depth))
            np.save("data/blocks/blocks_" + filename[:-4], blocks)
