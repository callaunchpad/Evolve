import numpy as np
import cv2
from image_utils import reconstruct
ims = np.load('test_ims.npy')
blocks = np.load('test_blocks.npy')
print(blocks[0].shape)
#x = np.reshape(blocks[0], (720, 1280, 3))
x = blocks[0][0][0]
print(x.shape)