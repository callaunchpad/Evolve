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

ap = argparse.ArgumentParser()
ap.add_argument("-H", "--rows", required=True,
    help="num rows")
ap.add_argument("-W", "--columns", required=True,
    help="num columns")
ap.add_argument("-D", "--depth", required=True,
    help="num channels")

ap.add_argument("-p", "--padding", required=False,default=0,
    help="padding setting (0 - zero, 1 - edge, 2 - reflect, 3 - Oleksii)")

args = vars(ap.parse_args())

blockRows = int(args["rows"])
blockColumns = int(args["columns"])
blockDepth = int(args["depth"])


def blockify_single_frame(filename,H=blockRows,W=blockColumns):

    img = cv2.imread("data/images/" + filename)

    #Actual height & width of image
    height = img.shape[0]
    width = img.shape[1]

    #Block Dimensions
    blockRows = H
    blockColumns = W

    #Padding setting
    padding = int(args['padding'])

    if padding == 0: #zero padding
        img2 = cv2.copyMakeBorder(img, 0, blockRows - height % blockRows, 0, blockColumns - width % blockColumns, cv2.BORDER_CONSTANT, (0,0,0,0))
    elif padding == 1: #edge copying padding
        img2 = cv2.copyMakeBorder(img, 0, blockRows - height % blockRows, 0, blockColumns - width % blockColumns, cv2.BORDER_REPLICATE)
    elif padding == 2: #edge reflection padding
        img2 = cv2.copyMakeBorder(img, 0, blockRows - height % blockRows, 0, blockColumns - width % blockColumns, cv2.BORDER_REFLECT)
    elif padding == 3: #oleksii padding
        img2 = img

    # cv2.imwrite(final, "data/blocks/_blocks" + filename)


    # Number of rows (Of Blocks)
    nRows = height//blockRows
    # Number of columns (Of Blocks)
    mCols = width//blockColumns

    #this for loop does not pad at the moment
    #added one to account for padding the rest'
    #idea: concatenate horizontally the distance from len - remainder the edge pixel
    if padding in range(3):
        final = []
        for i in range(0,nRows + 1):
            for j in range(0, mCols + 1):

                roi = img2[i*blockRows:i*blockRows + blockRows ,j*blockColumns:j*blockColumns + blockColumns]

                #cv2.imwrite("debug_output/"+str(frame_num)+'_blocks/block'+str(i)+str(j)+".jpg", roi)
                final.append(roi)

        np.save("data/blocks/blocks_" + filename,final)

    else:
        final = []
        for i in range(0,nRows + 1):
            for j in range(0, mCols + 1):
                if i == nRows and j == mCols:
                    roi = img[i*blockRows - (blockRows - (height % blockRows)):, j*blockColumns - (blockColumns - (width % blockColumns)):]
                elif i == nRows:
                    roi = img[i*blockRows - (blockRows - (height % blockRows)):, j*blockColumns:j*blockColumns + blockColumns]
                elif j == mCols:
                    roi = img[i*blockRows:i*blockRows + blockRows, j*blockColumns - (blockColumns - (width % blockColumns)):]
                else:
                    roi = img[i*blockRows:i*blockRows + blockRows, j*blockColumns:j*blockColumns + blockColumns]

                # cv2.imwrite("debug_output/"+str(frame_num)+'_blocks/block'+str(i)+str(j)+".jpg", roi)

                final.append(roi)
        np.save("data/blocks/blocks_" + filename,final)


if __name__ == "__main__":

    if not os.path.exists("data/blocks"):
        os.makedirs("data/blocks")

    if not os.path.exists("data/images"):
        os.makedirs("data/images")
        print("created images directory, please put in images")

    for filename in os.listdir("data/images"):
        if filename.endswith(".jpg"):
            blockify_single_frame(filename)
