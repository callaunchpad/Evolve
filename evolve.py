'''
usage: python evolve.py [-c (compress)] [-d (decompress)] [-v video.mp4] [-i image.png]
- links to specific codevector file

for compression:
- if image: return encode(image)
- if video: for each frame in video, append encode(frame) to compressed video
- encode(image) should return time it takes to encode
- generate "filename".ev (random file extension, can be whatever we want)

for decompression:
- return decode(code)
- decode(code) should return time it takes to decode
- generate "filename".[jpg/mp4]
note: later we'll merge compress and decompress into one exectuable so that we can just run "evolve -c image.jpg" for compression and "evolve -d code.ev" for decompression on the command-line
'''
# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from src.coding import *
import time
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument('-c', "--compress", action='store_true')
ap.add_argument('-d', "--decompress", action='store_true')
ap.add_argument("-v", "--video", required=False,default=None,
    help="video path")
ap.add_argument("-i", "--image", required=False,default=None,
    help="image path")

args = vars(ap.parse_args())

CODEVECTOR_PATH = "/path_to_codevector"

video_image_size = 0;

def run():
    if not args["compress"] and not args["decompress"]:
        print("missing argument -c or -d")
        raise ValueError('missing argument -c or -d"')
    if args["compress"] and args["decompress"]:
        print("cannot be both compress and decompress")
        raise ValueError('cannot be both compress and decompress')
    if args["video"] and args["image"]:
        print("cannot be both video and image")
        raise ValueError('cannot be both video and image')
    if not args["video"] and not args["image"]:
        print("need video or image")
        raise ValueError('need video or image')

    if args["compress"]:
        if args["image"]:
            tic = time.perf_counter()
            grey = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2GRAY)
            img_expanded = grey[:, :, np.newaxis]
            ret = encode(args["image"], img_expanded)
            toc = time.perf_counter()

            print(f"Encoded in {toc - tic:0.4f} seconds")
            #return ret #should be npy or ev?

            image = open(args["image"], "rb")
            f = image.read()
            b = bytearray(f)

            encoded_ev = open(args["image"] + ".ev", "rb")
            f2 = encoded_ev.read()
            b2 = bytearray(f2)

            print("Compression Ratio:" + str((len(b) * 8)/ len(b2)))




        elif args["video"]:
            tic = time.perf_counter()
            encoded_movie = [] #array of npys
            video_path = args["video"]
            i = 0
            v = cv2.VideoCapture(video_path)
            totsum = 0

            if not os.path.exists(args["video"].replace(".", "")):
                os.makedirs(args["video"].replace(".", ""))

            while True:
                ret, frame = v.read()
                if ret == True:
                    #b = cv2.resize(frame,(480, 360),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    img_expanded = grey[:, :, np.newaxis]
                    encode(args["video"].replace(".", "") + "/frame" + str(i).zfill(8), img_expanded)
                    temp = open(args["video"].replace(".", "") + "/frame" + str(i).zfill(8) + ".ev", "rb")
                    f2 = temp.read()
                    b2 = bytearray(f2)
                    totsum += len(b2)
                    i+=1
                else:
                    break
            toc = time.perf_counter()
            v.release()
            cv2.destroyAllWindows()

            print(f"Encoded in {toc - tic:0.4f} seconds")

            image = open(args["video"], "rb")
            f = image.read()
            b = bytearray(f)
            print("Compression Ratio:" + str((len(b)* 8) / totsum))



    elif args["decompress"]:

        if args["image"]:
            tic = time.perf_counter()
            ret = decode(args["image"])
            toc = time.perf_counter()

            print(f"Decoded in {toc - tic:0.4f} seconds")
            cv2.imwrite("decoded" + args["image"] + ".png", ret) #should be .jpg

        elif args["video"]:
            tic = time.perf_counter()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #(640, 360) is hardcoded, needs to be the size of the frame of the video
            out = cv2.VideoWriter("decoded_movie.mp4", fourcc, 30.0, (640,360), 0) #makes mp4
            #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            sorted = os.listdir(args["video"])
            sorted.sort()

            for ev_file in sorted:
                if ev_file.endswith(".ev"):
                    print(args["video"] + "/" + ev_file)
                    #needs proper format of uint8 for some reason
                    imagecopy = np.uint8(decode(args["video"] + "/" + ev_file))
                    out.write(imagecopy)
            #video_input = np.load("encoded_movie.ev")
            # for i in range(len(video_input)):
            #     out.write(decode(video_input[i]))

            toc = time.perf_counter()
            out.release()

            print(f"Decoded in {toc - tic:0.4f} seconds")

if __name__ == "__main__":
    run()
