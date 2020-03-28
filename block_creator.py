import cv2
import numpy as np
cam = cv2.VideoCapture('test.mp4')

count = 0
out_ims = []
out_blocks = []
while 1:
    ret, frame = cam.read()
    if ret:
        if count % 50 == 0:
            dst = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_ims.append(dst)
            block_dst = np.vsplit(dst, 9)
            # print(np.array(block_dst).shape)
            block_dst = [np.hsplit(bl, 16) for bl in block_dst]
            print(np.array(block_dst).shape)
            block_dst = np.reshape(block_dst, (144, 80, 80, 3))
            out_blocks.append(block_dst)
            # print(dst.shape)
        count = count + 1
    else:
        break

out_ims = np.array(out_ims)
out_blocks = np.array(out_blocks)
print(out_ims.shape)
print(out_blocks.shape)
np.save('test_ims.npy', out_ims)
np.save('test_blocks.npy', out_blocks)