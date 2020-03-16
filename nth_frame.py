import cv2
import numpy as np
import os

def process(vid_path, dim1, dim2, n):
    out = save_nth_frames(out, n, dim1, dim2)
    out2 = concat_all("./npy/", dim1, dim2)
    return out3

# crops video to dim1, dim2 and saves all frames
def save_nth_frames(vid_path, n, dim1, dim2):
    out = "./jpg/" + vid_path.split('.')[0] + "_every_" + str(n) + "_at_"
    v = cv2.VideoCapture(vid_path)
    i = 0
    images = []
    while (v.isOpened()):
        if i % n != 0:
            break
        ret, frame = v.read()
        if ret == False:
            break
        b = cv2.resize(frame,(dim1, dim2),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(out + str(i) + ".jpg", b)


        np_frame = cv2.imread(out + str(i) + ".jpg", 1)
        images.append(np_frame)
    all_frames = np.array(images)

    out2 = "./npy/" + vid_path.split('.')[0] + "_every_" + str(n)
    np.save(out2, all_frames)

    v.release()
    cv2.destroyAllWindows

    return

def concat_all(path, dim1, dim2):
    dir = os.listdir(path)
    arrays = []
    for file in dir:
        if (file.split('.')[1] != ".npy"):
            break
        arr = np.load(file)
        arrays.append(arr)

    concat = np.concatenate([arrays])
    np.save("all_frames", concat)
    return "all_frames"
