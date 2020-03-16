from dahuffman import HuffmanCodec
import collections
import numpy as np

class HuffmanEncoder:
    #Assuming dataset is a list of numpy integer sequences. 
    def __init__(self, dataset):
        c = collections.Counter()
        for row in dataset: 
            c.update(row)
        self.collection = c
        print(self.collection)
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