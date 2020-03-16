# USAGE
# python3 blockify.py -i input.jpg -H 100 -W 100 -p 1

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

blockRows = int(args["rows"])
blockColumns = int(args["columns"])

output = [] # each entry being one block, (shape should be num blocks per frame x num frames)


def blockify_single_frame(frame_num,img,H=blockRows,W=blockColumns):

	global output

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

	if not os.path.exists("debug_output/"+str(frame_num)+'_blocks'):
		os.makedirs("debug_output/"+str(frame_num)+'_blocks')

	cv2.imwrite("debug_output/"+str(frame_num)+"_blocks/_resized_img.jpg",img2)

	# Number of rows (Of Blocks)
	nRows = height//blockRows
	# Number of columns (Of Blocks)
	mCols = width//blockColumns

	#this for loop does not pad at the moment
	#added one to account for padding the rest'
	#idea: concatenate horizontally the distance from len - remainder the edge pixel
	if padding in range(3):
		for i in range(0,nRows + 1):
			for j in range(0, mCols + 1):

				roi = img2[i*blockRows:i*blockRows + blockRows ,j*blockColumns:j*blockColumns + blockColumns]

				cv2.imwrite("debug_output/"+str(frame_num)+'_blocks/block'+str(i)+str(j)+".jpg", roi)
				output.append(roi)

	else:
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

				cv2.imwrite("debug_output/"+str(frame_num)+'_blocks/block'+str(i)+str(j)+".jpg", roi)
				output.append(roi)

if __name__ == "__main__":

	"""
	#generate dummy movie of 480x360 images 10 frames long
	dummy_movie = []
	for i in range(10):
		dummy_movie.append((np.random.standard_normal([360, 480, 3]) * 255).astype(np.uint8))
	np.save("dummy_movie",dummy_movie)
	"""

	#load dummy video and resize to 480x360 and save as .npy
	dummy_movie = []
	v = cv2.VideoCapture("omae_wa_mou.mp4")
	while True:
		ret, frame = v.read()
		if ret == True:
			b = cv2.resize(frame,(480, 360),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
			dummy_movie.append(b)
		else:
			break
	v.release()
	cv2.destroyAllWindows()
	np.save("input_movie",dummy_movie)

	#loading in video input
	video_input = np.load("input_movie.npy")

	for i in range(len(video_input)):
		blockify_single_frame(i,video_input[i])

	print(np.asarray(output).shape) #verify output shape
	np.save("blockified_output",output)
