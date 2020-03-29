# USAGE
# python3 some_shit_that_splits.py -i "input.jpg" -n 100 -m 100


# import the necessary packages
#from keras.preprocessing.image import img_to_array
#from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import sys

global output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_name", required=True,
	help="image name")
ap.add_argument("-r", "--rows", required=True,
	help="num rows")
ap.add_argument("-c", "--columns", required=True,
	help="num columns")
ap.add_argument("-p", "--padding", required=True,
	help="padding setting (0 - zero, 1 - edge, 2 - reflect, other - Oleksii)")

output = []

args = vars(ap.parse_args())

def split():
	img = cv2.imread(args["image_name"])

	#Actual height & width of image
	height = img.shape[0]
	width = img.shape[1]

	#Patch Dimensions
	patchRows = int(args["rows"])
	patchColumns = int(args["columns"])

	#Padding setting
	padding = int(args["padding"])

	if padding == 0:
		img2 = cv2.copyMakeBorder(img, 0, patchRows - height % patchRows, 0, patchColumns - width % patchColumns, cv2.BORDER_CONSTANT, (0,0,0,0))
	elif padding == 1:
		img2 = cv2.copyMakeBorder(img, 0, patchRows - height % patchRows, 0, patchColumns - width % patchColumns, cv2.BORDER_REPLICATE)
	elif padding == 2:
		img2 = cv2.copyMakeBorder(img, 0, patchRows - height % patchRows, 0, patchColumns - width % patchColumns, cv2.BORDER_REFLECT)
	else:
		img2 = img

	#cv2.imshow("Image2", img2)
	cv2.imwrite("patches/img2.jpg",img2)

	# Number of rows (Of Patches)
	nRows = height//patchRows
	# Number of columns (Of Patches)
	mCols = width//patchColumns

	print(img.shape)

	#this for loop does not pad at the moment
	#added one to account for padding the rest'
	#idea: concatenate horizontally the distance from len - remainder the edge pixel
	if padding in range(3):
		for i in range(0,nRows + 1):
			for j in range(0, mCols + 1):

				roi = img2[i*patchRows:i*patchRows + patchRows ,j*patchColumns:j*patchColumns + patchColumns]
				output.append(roi)
				#cv2.imshow('rois'+str(i)+str(j), roi)
				cv2.imwrite('patches/patch_'+str(i)+str(j)+".jpg", roi)

	else:
		for i in range(0,nRows + 1):
			for j in range(0, mCols + 1):
				if i == nRows and j == mCols:
					roi = img[i*patchRows - (patchRows - (height % patchRows)):, j*patchColumns - (patchColumns - (width % patchColumns)):]
				elif i == nRows:
					roi = img[i*patchRows - (patchRows - (height % patchRows)):, j*patchColumns:j*patchColumns + patchColumns]
				elif j == mCols:
					roi = img[i*patchRows:i*patchRows + patchRows, j*patchColumns - (patchColumns - (width % patchColumns)):]
				else:
					roi = img[i*patchRows:i*patchRows + patchRows, j*patchColumns:j*patchColumns + patchColumns]
					output.append(roi)
				#cv2.imshow('rois'+str(i)+str(j), roi)
				cv2.imwrite('patches/patch_'+str(i)+str(j)+".jpg", roi)



	print("done.")

if __name__ == "__main__":
	if not os.path.exists('patches'):
		os.makedirs('patches')

	split()
	print(np.asarray(output).shape) #verify output shape
	np.save("output_array",output)
