import cv2
import numpy as np
import os

"""
This class will save every nth frame of a specific video, cropped to desired dimensions,
and concatenate the frames as an .npy file.

Parameters:

self.vid_path -- the path at which the video to be processed is located (e.g. "./nature.mp4")
vid_name -- the self.vid_name of the video, which will appear in the output npy file (e.g. "nature")
self.dim1, self.dim2 -- the height, width to crop each frame to (so it has shape self.dim1 x self.dim2 x 3)
self.n -- such that each nth frame is saved (default 20)

""""""

Usage:

After instantiating a VideoProcessor, call process(). Two folders will be created.
The one ending in "jpg" will contain jpg images of each frame saved; the one ending in "npy" contains
a concatenation of all frames as npy files.
The class variable images will contain each frame, and images_npy will contain each frame as a npy file.
"""

class VideoProcessor:

    def __init__(self, vid_path, vid_name, dim1, dim2, n=20):

        self.images = []
        self.images_npy = None
        self.vid_name = vid_name
        self.vid_path = vid_path
        self.dim1 = dim1
        self.dim2 = dim2
        self.n = n

    # crops video to self.dim1, self.dim2 and saves all frames
    def process(self):
        jpg_folder = "./" + self.vid_name + "_every_" + str(self.n) + "_" + "jpg"
        os.mkdir(jpg_folder)
        v = cv2.VideoCapture(self.vid_path)
        i = 0
        self.images = []

        while (v.isOpened()):
            ret, frame = v.read()

            # video is done
            if not ret:
                total_frames = v.get(cv2.CAP_PROP_FRAME_COUNT)
                print("Video file finished. Total Frames: %d" % (total_frames))
                print("Frames Saved: %d" % (total_frames // self.n) )
                break

            # save this frame
            if i % self.n == 0:
                b = frame[0:self.dim1, 0:self.dim2].copy()
                out = jpg_folder + "/" + "frame" + str(i) + ".jpg"
                cv2.imwrite(out, b)
                np_frame = cv2.imread(out, 1)
                print(np_frame.shape)
                self.images.append(np_frame)

            i = i + 1

            # timeout handler
            key = cv2.waitKey(25)
            if key == ord('q'):
                break

        v.release()
        cv2.destroyAllWindows

        images_to_npy()
        return

    # saves the images array as a concatenated npy file
    def images_to_npy(self):
        npy_folder = "./" + self.vid_name + "_every_" + str(self.n) + "_" + "npy"
        os.mkdir(npy_folder)

        self.images_npy = np.asarray(self.images)

        out2 = npy_folder +  "/images_npy"
        np.save(out2, self.images_npy)
