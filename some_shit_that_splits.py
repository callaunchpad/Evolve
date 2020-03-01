# USAGE
# python classify.py --model apple_not_apple.model

# import the necessary packages
#from keras.preprocessing.image import img_to_array
#from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_name", required=True,
	help="image name")
ap.add_argument("-n", "--rows", required=True,
	help="num rows")
ap.add_argument("-m", "--columns", required=True,
	help="num columns")
args = vars(ap.parse_args())

def split():
	img = cv2.imread(args["image_name"])
	height = img.shape[0]
	width = img.shape[1]


	#number of rows
	nRows = height//int(args["rows"])
	# Number of columns
	mCols = width//int(args["columns"])

	print(img.shape)
	
	for i in range(0,nRows):
		for j in range(0, mCols):
			roi = img[i*height//nRows:i*height//nRows + height//nRows ,j*width//mCols:j*width//mCols + width//mCols]
			#cv2.imshow('rois'+str(i)+str(j), roi)
			cv2.imwrite('patches/patch_'+str(i)+str(j)+".jpg", roi)

	print("done.")

if __name__ == main:
	split()
