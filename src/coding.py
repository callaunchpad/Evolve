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

class HuffmanEncoder:
    # dataset is a sequence of numpy integers
    def __init__(self, dataset):
        # self.input = np.array_str(dataset)[1:-1]
        self.items = [(str(i), j) for i,j in sorted(collections.Counter(dataset).items())]
        self.codebook = huffman.codebook(self.items)
    # call this to print codebook to a specific path
    def print_codebook(self, path):
        f = open(path,"w+")
        for item in self.codebook.items():
            f.write(str(item[0]) + " " + str(item[1]))
            f.write("\n")
        f.close()
 # encoder = HuffmanEncoder(codevector)

#ret = encode(args["image"])

#ingests some string
def encode(image):
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
    CODEVECTOR_PATH = "src/small.pngcodevectors.npy"
    codevector = np.load(CODEVECTOR_PATH)
    flattened_blocks = blockify(image, (80,80,3))
    result = []

    #first line of text file = image size
    result.append(np.shape(image))

    best = np.array([])
    flat_codevector = np.reshape(codevector, (codevector.shape[0], codevector.shape[1] * codevector.shape[2] * codevector.shape[3])).T
    for block in flattened_blocks:
        best = np.append(best, [closest_codeblock_index(flat_codevector, block)])

    best = best.astype(int)
    print(best.shape)
    print(best)
    encoder = HuffmanEncoder(best)

    encoded = np.array([])

    # pop whitespace key
    encoder.codebook.pop(" ", None)
    print(encoder.codebook)
    for i in best:
        encoded = np.append(encoded, encoder.codebook[str(i)])

    #second line of text file = codevectors
    result.append(encoded)

    # index 0 = image
    # index 1 = vals
    # index 2 = tree
    result.append(encoder.codebook)
    np.save("encoded.ev", result);

#USAGE: decode(args["image"])
def decode(ev_path):
    CODEVECTOR_PATH = "src/small.pngcodevectors.npy"
    codevector = np.load(CODEVECTOR_PATH)
    print(codevector.shape)

    input = np.load(ev_path, allow_pickle = True)
    codebook = input[2] #tree
    encodes = input[1] #vals
    print(type(encodes))
    image_size = input[0] #img

    blocks = []
    print(encodes)
    print(codebook)
    print(codebook.values())

    for encode in encodes:
        for img, code in codebook.items():
            if code == encode:
                print(img)
                blocks.append(codevector[int(img)])
    blocks = np.asarray(blocks)
    print(blocks.shape)
    decoded = reconstruct_image(blocks, image_size)
    return decoded
