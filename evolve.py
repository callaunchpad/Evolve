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

ap = argparse.ArgumentParser()
ap.add_argument('-c', "--compress", action='store_true')
ap.add_argument('-d', "--decompress", action='store_true')
ap.add_argument("-v", "--video", required=False,default=None,
    help="video path")
ap.add_argument("-i", "--image", required=False,default=None,
    help="image path")

args = vars(ap.parse_args())

CODEVECTOR_PATH = "/path_to_codevector"

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
            ret = encode(cv2.imread(args["image"]))
            toc = time.perf_counter()

            print(f"Encoded in {toc - tic:0.4f} seconds")
            #return ret #should be npy or ev?

        elif args["video"]:
            tic = time.perf_counter()
            encoded_movie = [] #array of npys
            video_path = args["video"]
            v = cv2.VideoCapture(video_path)
            while True:
                ret, frame = v.read()
                if ret == True:
                    #b = cv2.resize(frame,(480, 360),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                    encoded_movie.append(encode(frame))
                else:
                    break
            toc = time.perf_counter()

            v.release()
            cv2.destroyAllWindows()

            print(f"Encoded in {toc - tic:0.4f} seconds")

            np.save("encoded_movie.ev", encoded_movie)


    elif args["decompress"]:

        if args["image"]:
            tic = time.perf_counter()
            ret = decode(args["image"])
            toc = time.perf_counter()

            print(f"Decoded in {toc - tic:0.4f} seconds")
            cv2.imwrite("decoded.jpg", ret) #should be .jpg

        elif args["video"]:
            tic = time.perf_counter()
            fourcc = VideoWriter_fourcc(*'MP4V')
            out = VideoWriter("decoded_movie.mp4", fourcc, 30.0, (640,480)) #makes mp4
            #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

            video_input = np.load("encoded_movie.ev")
            for i in range(len(video_input)):
                out.write(decode(video_input[i]))

            toc = time.perf_counter()
            out.release()

            print(f"Decoded in {toc - tic:0.4f} seconds")

if __name__ == "__main__":
    run()
