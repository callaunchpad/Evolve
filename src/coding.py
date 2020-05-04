'''
- contains encode and decode functions
- will have huffman encoding functionality
'''

# requires huffman package
import huffman
# from dahuffman import HuffmanCodec
import collections
import numpy as np
import cv2
from src.utils import *

#huffman = True -> a sequence of bits
#huffman = True, delimeter = True -> sequence of bits with spaces
#huffman = False, delimeter = either -> numerical indices with spaces in between

class HuffmanEncoder:
    # dataset is a sequence of numpy integers
    def __init__(self, dataset):
        # self.input = np.array_str(dataset)[1:-1]
        self.items = [(str(i), j) for i,j in sorted(collections.Counter(dataset).items())]
        self.codebook = huffman.codebook(self.items)

        self.codebook.pop(" ", None)
        self.codebook.pop("/n", None)
        print(self.codebook)
    # call this to print codebook to a specific path
    def print_codebook(self, path):
        f = open(path,"w+")
        for item in self.codebook.items():
            f.write(str(item[1]))
            f.write("\n")
        f.close()
    def write_file_ouptut(self, path, best, image_shape, _huffman, _delimiter, codevector):
        f = open(path,"w+")

        # writing image_name (100,100,3)
         #01010101
        f.write(image_helper(image_shape[0], 16))
        f.write(image_helper(image_shape[1], 16))
        f.write(image_helper(image_shape[2], 2))
        f.write("\n")

        # writing mapping
        for i in range(len(codevector)):
            if (str(i) in self.codebook.keys()):
                f.write(str(self.codebook[str(i)]))
                f.write("\n")
            else:
                f.write("\n")

        for index_value in best: #[0,1,2,3,4,5,6,7,8,9]
            if _delimiter == True:
                f.write(str(self.codebook[str(index_value)])) #0001 001
                f.write(" ")
            else:
                f.write(str(self.codebook[str(index_value)])) #0001001
        f.close()
 # encoder = HuffmanEncoder(codevector)

#ret = encode(args["image"])
def image_helper(image_size, bits):
    print(image_size)
    if bits > 8:
        temp = '{0:016b}'.format(image_size)
        print(temp)
        return temp
    else:
        temp = '{0:02b}'.format(image_size)
        print(temp)
        return temp

CODEVECTOR_PATH = "src/blocks_download3.npy" #put codevector path here

#ingests some string
def encode(image_name, image, _huffman = True, _delimeter = False):
    global CODEVECTOR_PATH

    codevector = np.load(CODEVECTOR_PATH)
    flattened_blocks = blockify(image, codevector.shape[1:])

    best = np.array([])
    flat_codevector = np.reshape(codevector, (codevector.shape[0], codevector.shape[1] * codevector.shape[2] * codevector.shape[3])).T
    for block in flattened_blocks:
        best = np.append(best, [closest_codeblock_index(flat_codevector, block)])

    best = best.astype(int)
    if _huffman:
        encoder = HuffmanEncoder(best)
        #encoder.write_file_ouptut("encoded.ev", best, image.shape, _huffman, _delimeter, codevector)
        encoder.write_file_ouptut(image_name+".ev", best, image.shape, _huffman, _delimeter, codevector)
    else:
        f = open(image_name+".ev","w+")

        # writing image_name
        f.write(image_helper(image.shape[0], 16))
        f.write(image_helper(image.shape[1], 16))
        f.write(image_helper(image.shape[2], 2))
        f.write("\n")

        for index_value in best:
            f.write(str(index_value)) #1 2 3
            f.write(" ")

        f.close()

#USAGE: decode(args["image"])
def decode(ev_path, _huffman = True, _delimeter = False):
    global CODEVECTOR_PATH

    codevector = np.load(CODEVECTOR_PATH)

    f = open(ev_path, "r")
    lines = f.readlines()
    print(codevector.shape)

    image_size_line = lines[0]
    x = int(image_size_line[0:16], 2)
    y = int(image_size_line[16:32], 2)
    z = int(image_size_line[32:34], 2)
    image_size = (x, y, z)
    codebook = {}
    blocks = []

    if _huffman:
        for i in range(len(lines[1:-1])):
            codebook[i] = lines[i + 1][:-1]
        print(codebook)
        last_line = lines[-1]

        if _delimeter:
            encoded_values = last_line.split(" ") #["01", "01", "001"]
            for encode in encoded_values:
                for img, code in codebook.items():
                    if code == encode:
                        blocks.append(codevector[int(img)])
        else:
            candidate = ""
            for char in last_line:
                candidate+=char
                for img, code in codebook.items():
                    if code == candidate:
                        #do addint stuff here
                        blocks.append(codevector[int(img)])
                        candidate = ""
                        break
    else:
        last_line = lines[-1]
        encoded_values = last_line.split(" ")
        encoded_values.remove("")
        blocks = [codevector[int(i)] for i in encoded_values]

    blocks = np.asarray(blocks)

    decoded = reconstruct_image(blocks, image_size)
    return decoded
