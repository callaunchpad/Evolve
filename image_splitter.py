# USAGE
# python3 image_splitter.py -i input.jpg -H 100 -W 100 -p 1

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_name", required=False,default="input.jpg",
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

output = [] # each entry being one patch, (shape should be num patches per frame x num frames)


def split_single_frame(frame_num,img,H=patchRows,W=patchColumns):

	global output

	#Actual height & width of image
	height = img.shape[0]
	width = img.shape[1]

	#Patch Dimensions
	patchRows = H
	patchColumns = W

	#Padding setting
	padding = int(args['padding'])

	if padding == 0: #zero padding
		img2 = cv2.copyMakeBorder(img, 0, patchRows - height % patchRows, 0, patchColumns - width % patchColumns, cv2.BORDER_CONSTANT, (0,0,0,0))
	elif padding == 1: #edge copying padding
		img2 = cv2.copyMakeBorder(img, 0, patchRows - height % patchRows, 0, patchColumns - width % patchColumns, cv2.BORDER_REPLICATE)
	elif padding == 2: #edge reflection padding
		img2 = cv2.copyMakeBorder(img, 0, patchRows - height % patchRows, 0, patchColumns - width % patchColumns, cv2.BORDER_REFLECT)
	elif padding == 3: #oleksii padding
		img2 = img

	if not os.path.exists(str(frame_num)+'_patches'):
		os.makedirs(str(frame_num)+'_patches')

	cv2.imwrite(str(frame_num)+"_patches/_resized_img.jpg",img2)

	# Number of rows (Of Patches)
	nRows = height//patchRows
	# Number of columns (Of Patches)
	mCols = width//patchColumns

	#this for loop does not pad at the moment
	#added one to account for padding the rest'
	#idea: concatenate horizontally the distance from len - remainder the edge pixel
	if padding in range(3):
		for i in range(0,nRows + 1):
			for j in range(0, mCols + 1):

				roi = img2[i*patchRows:i*patchRows + patchRows ,j*patchColumns:j*patchColumns + patchColumns]

				cv2.imwrite(str(frame_num)+'_patches/patch_'+str(i)+str(j)+".jpg", roi)
				output.append(roi)



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

				cv2.imwrite(str(frame_num)+'_patches/patch_'+str(i)+str(j)+".jpg", roi)
				output.append(roi)

if __name__ == "__main__":
	
	#movie of 480x360 images 10 frames long
	random_movie = []
	for i in range(10):
		random_movie.append((np.random.standard_normal([360, 480, 3]) * 255).astype(np.uint8))
	np.save("random_movie",random_movie)

	#loading in video input
	video_input = np.load("random_movie.npy")

	for i in range(len(video_input)):
		split_single_frame(i,video_input[i])

	print(np.asarray(output).shape) #verification
	np.save("splitted_output",output)

