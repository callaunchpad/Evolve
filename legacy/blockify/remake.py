# USAGE
# python3 remake.py -i input.jpg -p 1

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_name", required=False,default="input.jpg",
    help="image name")
ap.add_argument("-p", "--padding", required=False,default=0,
    help="padding setting (0 - zero, 1 - edge, 2 - reflect, 3 - Oleksii)")

args = vars(ap.parse_args())


#(num_centroids, block_width, block_height, 3)
# read_array = np.random.randn(30, 100, 100, 3)*255
read_array = np.load("output_array.npy")

output = [] # each horizontal row

def remake_frame(img,H=len(read_array[1]),W=len(read_array[2])):

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

    if not os.path.exists("debug_output/"+'test'):
        os.makedirs("debug_output/"+'test')

    cv2.imwrite("debug_output/"+"test/_resized_img.jpg",img2)

    # Number of rows (Of Blocks)
    nRows = height//blockRows
    # Number of columns (Of Blocks)
    mCols = width//blockColumns

    #this for loop does not pad at the moment
    #added one to account for padding the rest'
    #idea: concatenate horizontally the distance from len - remainder the edge pixel
    if padding in range(3):
        for i in range(0,nRows + 1):
            rowArray = []
            for j in range(0, mCols + 1):

                roi = img2[i*blockRows:i*blockRows + blockRows ,j*blockColumns:j*blockColumns + blockColumns]

                best = min(read_array, key = lambda x: mse(roi, x))

                print(np.where(read_array == best))

                cv2.imwrite("debug_output/"+'test/block'+str(i)+str(j)+".jpg", best)

                rowArray.append(best)

            for k in range(1, len(rowArray)):
                rowArray[k] = np.concatenate((rowArray[k-1],rowArray[k]), axis = 1)

            output.append(rowArray[len(rowArray)-1])

        for z in range(1, len(output)):
            output[z] = np.concatenate((output[z-1], output[z]),  axis = 0)
        cv2.imwrite("debug_output/final.jpg", output[len(output)-1])

    else:
        for i in range(0,nRows + 1):
            rowArray = []
            pic = []
            for j in range(0, mCols + 1):
                if i == nRows and j == mCols:
                    roi = img[i*blockRows - (blockRows - (height % blockRows)):, j*blockColumns - (blockColumns - (width % blockColumns)):]
                elif i == nRows:
                    roi = img[i*blockRows - (blockRows - (height % blockRows)):, j*blockColumns:j*blockColumns + blockColumns]
                elif j == mCols:
                    roi = img[i*blockRows:i*blockRows + blockRows, j*blockColumns - (blockColumns - (width % blockColumns)):]
                else:
                    roi = img[i*blockRows:i*blockRows + blockRows, j*blockColumns:j*blockColumns + blockColumns]

                best = min(read_array, key = lambda x: mse(roi, x))

                print(np.where(read_array == best))

                cv2.imwrite("debug_output/"+'test/block'+str(i)+str(j)+".jpg", best)

                rowArray.append(best)

            for k in range(1, len(rowArray)):
                rowArray[k] = np.concatenate((rowArray[k-1],rowArray[k]), axis = 1)


            output.append(rowArray[len(rowArray)-1])

        for z in range(1, len(output)):
            output[z] = np.concatenate((output[z-1], output[z]),  axis = 0)
        cv2.imwrite("debug_output/final.jpg", output[len(output)-1])

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

if __name__ == "__main__":

    """
    #generate dummy movie of 480x360 images 10 frames long
    dummy_movie = []
    for i in range(10):
        dummy_movie.append((np.random.standard_normal([360, 480, 3]) * 255).astype(np.uint8))
    np.save("dummy_movie",dummy_movie)
    """
    input_image = cv2.imread(args["image_name"])

    remake_frame(input_image)

    print(np.asarray(output).shape) #verify output shape
    np.save("blockified_output",output)
