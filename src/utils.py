import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'
import math
import numpy as np
from keras_contrib.losses import DSSIMObjective
import tensorflow as tf
from keras import backend as K

'''
- contain code for reconstruct, blockify, finding nearest codevector
'''

ssim = lambda im1, im2: 1 - K.get_value(DSSIMObjective(kernel_size=3).__call__(tf.cast(np.array(im1), tf.float32), tf.cast(np.array(im2), tf.float32)))

def reconstruct_image(blocks, image_size):
    image = np.zeros(image_size)
    avg = np.zeros(image_size)
    bh = blocks.shape[1]
    bw = blocks.shape[2]
    bd = blocks.shape[3]
    for i in range(blocks.shape[0]):
        fitH = math.ceil(image_size[0]/bh) #2
        overH = image_size[0]%bh #2
        fitW = math.ceil(image_size[1]/bw) #2
        overW = image_size[1]%bw #2

        h0 = image_size[0]-bh if bh*(i//fitW)+bh>image_size[0] else bh*(i//fitW)
        h1 = h0+bh
        w0 = image_size[1]-bw if bw*(i%fitW)+bw>image_size[1] else bw*(i%fitW)
        w1 = w0+bw

        avg[h0:h1,w0:w1,:] += np.ones((bh, bw, bd))
        image[h0:h1,w0:w1,:] += blocks[i]
    return np.divide(image,avg)

def closest_codeblock_index(flattened_cb, bl):
    bl_r = np.reshape(bl, (-1, 1))
    norm = np.linalg.norm(flattened_cb - bl_r, axis=0)
    return np.argmin(norm, axis=0)