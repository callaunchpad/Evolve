import cv2
import numpy as np
cam = cv2.VideoCapture('test.mp4')

block_size = 5
n_channels = 1

count = 0
out_ims = []
out_blocks = []
while 1:
    ret, frame = cam.read()
    if ret:
        if count % 50 == 0:
            if n_channels == 1:
                dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                dst = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_ims.append(dst)
            block_dst = np.vsplit(dst, dst.shape[0] // block_size)
            block_dst = [np.hsplit(bl, dst.shape[1] // block_size) for bl in block_dst]
            print(str(count))
            block_dst = np.reshape(block_dst, ((dst.shape[1] * dst.shape[0]) // (block_size ** 2), block_size, block_size, 1))
            out_blocks.append(block_dst)
        count = count + 1
    else:
        break

print(np.array(out_ims).shape)
print(np.array(out_blocks).shape)
out_ims = np.expand_dims(np.array(out_ims), axis = 3)
out_blocks = np.array(out_blocks)
print(out_ims.shape)
print(out_blocks.shape)
np.save('test_ims.npy', out_ims)
np.save('test_blocks.npy', out_blocks)