# USAGE
# python3 image_splitter.py -i input.jpg -n 100 -m 100 -p 0


# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os
import sys

#np.save("output,npz",a=obj)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_name", required=True,
	help="image name")
ap.add_argument("-H", "--rows", required=True,
	help="num rows")
ap.add_argument("-W", "--columns", required=True,
	help="num columns")
ap.add_argument("-p", "--padding", required=False,default=0,
	help="padding setting (0 - zero, 1 - edge, 2 - reflect, 3 - Oleksii)")

args = vars(ap.parse_args())

patchRows = int(args["rows"])
patchColumns = int(args["columns"])

def split(H=patchRows,W=patchColumns):
	img = cv2.imread(args["image_name"])

	#Actual height & width of image
	height = img.shape[0]
	width = img.shape[1]

	#Patch Dimensions
	patchRows = H
	patchColumns = W

	#Padding setting
	padding = int(args['padding'])

	if padding == 0:
		img2 = cv2.copyMakeBorder(img, 0, patchRows - height % patchRows, 0, patchColumns - width % patchColumns, cv2.BORDER_CONSTANT, (0,0,0,0))
	elif padding == 1:
		img2 = cv2.copyMakeBorder(img, 0, patchRows - height % patchRows, 0, patchColumns - width % patchColumns, cv2.BORDER_REPLICATE)
	elif padding == 2:
		img2 = cv2.copyMakeBorder(img, 0, patchRows - height % patchRows, 0, patchColumns - width % patchColumns, cv2.BORDER_REFLECT)
	elif padding == 3:
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

				#cv2.imshow('rois'+str(i)+str(j), roi)
				cv2.imwrite('patches/patch_'+str(i)+str(j)+".jpg", roi)


	print("done.")

if __name__ == "__main__":

	random_input = np.random.rand(10,1920,1080)*255 #10 frames of 1920x1080 input image
	np.save("random_input",random_input)
	img = np.load("random_input.npy")

	if not os.path.exists('patches'):
		os.makedirs('patches')

	split()