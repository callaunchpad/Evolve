'''
- contains encode and decode functions
- will have huffman encoding functionality
'''

from dahuffman import HuffmanCodec
import collections
import numpy as np
import cv2
import utils

CODEVECTOR_PATH = "/path_to_codevector"
codevector = np.load(CODEVECTOR_PATH)

encoder = HuffmanEncoder(codevector)

#ret = encode(args["image"])

#ingests some string
def encode(image):
    #pass
    # block yea im writing it out here just for trolls we can move it
    flattened_blocks = utils.blockify(image)
    result = []
    encoded = []
    for block in flattened_blocks:
        closest = closest_codeblock_index(codevector, block)
        encoded.append(encoder.encode(closest))
    result.append(encoded)

    # index 0 = encoded stuff
    # index 1 = image resize
    # index 2 = tree
    result.append(np.shape(image))
    result.append(encoder)
    np.save("encoded.ev", result);

#USAGE: decode(args["image"])
def decode(ev_path):
    input = np.load("encoded.ev")
    encoder = input.pop() #tree
    image_shape = input.pop() #tree
    encodes = input.pop()

    blocks = []
    for encode in encodes:
        blocks.append(encoder.decode(encode))

    decoded = reconstruct_image(blocks, image_size)
    return decoded
    pass


class HuffmanEncoder:
    #Assuming dataset is a list of numpy integer sequences.
    def __init__(self, dataset):
        mapping = {}
        c = collections.Counter()
        for row in dataset:
            c.update(row)
        self.collection = c
        self.codec = HuffmanCodec.from_frequencies(self.collection)

    #Returns bytes object, looks like a string random characters.
    def encode(self, seq):
        return self.codec.encode(seq)

    #Takes in bytes object.
    def decode(self, string):
        return self.codec.decode(string)

    #Assume 16 bits per integer.
    def input_len(self, seq):
        return len(seq) * 16

    #Each char is 8 bits.
    def encoded_len(self, encoding):
        return len(encoding)*8
