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
    if bits > 8:
        temp = "00000000" + bin(image_size)[2:]
        return temp
    else:
        temp = bin(image_size)[2:]
        return temp



#ingests some string
def encode(image, _huffman = True, _delimeter = False):
    #pass
    # block yea im writing it out here just for trolls we can move it
    # CODEVECTOR_PATH = "src/download3_jpgcodevectors.npy"
    # codevector = np.load(CODEVECTOR_PATH)
    # flattened_blocks = blockify(image, (100,100,1))
    # result = []
    # encoded = []
    # flat_codevector = np.reshape(codevector, (codevector.shape[0], codevector.shape[1] * codevector.shape[2] * codevector.shape[3])).T
    # for block in flattened_blocks:
    #     closest = codevector[(closest_codeblock_index(flat_codevector, block))]
    #     print(encoder.codebook)
    #     encoded.append(encoder.codebook[closest])
    # result.append(encoded)


    CODEVECTOR_PATH = "src/blocks_input_codevector.npy"
    codevector = np.load(CODEVECTOR_PATH)
    print(codevector.shape[1:])
    flattened_blocks = blockify(image, codevector.shape[1:])
    # result = []
    #
    # #first line of text file = image size
    # result.append(np.shape(image))

    best = np.array([])
    flat_codevector = np.reshape(codevector, (codevector.shape[0], codevector.shape[1] * codevector.shape[2] * codevector.shape[3])).T
    for block in flattened_blocks:
        best = np.append(best, [closest_codeblock_index(flat_codevector, block)])

    best = best.astype(int)
    print(best.shape)
    print(best)
    if _huffman:
        encoder = HuffmanEncoder(best)
        encoder.write_file_ouptut("encoded.ev", best, image.shape, _huffman, _delimeter, codevector)
    else:
        f = open(path,"w+")

        # writing image_name
        f.write(image.shape)
        f.write("\n")

        for index_value in best:
            f.write(str(index_value)) #1 2 3
            f.write(" ")

        f.close()
    # encoded = np.array([])
    #
    # # pop whitespace key
    # encoder.codebook.pop(" ", None)
    # print(encoder.codebook)
    # for i in best:
    #     encoded = np.append(encoded, encoder.codebook[str(i)])
    #
    # #second line of text file = codevectors
    # result.append(encoded)

    # index 0 = image
    # index 1 = vals
    # index 2 = tree
    # result.append(encoder.codebook)
    # np.save("encoded.ev", result);

#USAGE: decode(args["image"])
def decode(ev_path, _huffman = True, _delimeter = False):
    CODEVECTOR_PATH = "src/blocks_input_codevector.npy"
    codevector = np.load(CODEVECTOR_PATH)
    print(codevector.shape)

    f = open(ev_path, "r")
    lines = f.readlines()

    image_size_line = lines[0]
    x = int(image_size_line[0:16], 2)
    y = int(image_size_line[16:32], 2)
    z = int(image_size_line[32:34], 2)
    image_size = (x, y, z)
    codebook = {}
    blocks = []
    print(image_size)

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
    else:
        last_line = lines[-1]
        encoded_values = last_line.split(" ")
        blocks = [int(i) for i in encoded_values]

    # input = np.load(ev_path, allow_pickle = True)
    # codebook = input[2] #tree
    # encodes = input[1] #vals
    # print(type(encodes))
    # image_size = input[0] #img
    #
    # blocks = []
    # print(encodes)
    # print(codebook)
    # print(codebook.values())
    #
    # for encode in encodes:
    #     for img, code in codebook.items():
    #         if code == encode:
    #             print(img)
    #             blocks.append(codevector[int(img)])
    blocks = np.asarray(blocks)
    print(blocks.shape)
    decoded = reconstruct_image(blocks, image_size)
    return decoded
