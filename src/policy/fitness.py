import sys
sys.path.append('..')

import numpy as np
from utils import reconstruct_image, closest_codeblock_index
'''
fitness/selection/crossover/mutation.py
- abstract away the policies into respective files (for ease of experimentation)
- implement gradient mutation + single-point set crossover, 2x2x1 block size
- later: nested vector quantization, sepearate channel quantization
'''

def ssim_reconstruct(cb, train_im, train_blocks, batch_size = 1):
    used_indeces = np.random.choice(train_im.shape[0], batch_size)
    fitness_val_per_im = []
    
    mat = np.reshape(cb,
        (cb.shape[0], cb.shape[1] * cb.shape[2] * cb.shape[3])).T
    
    ground_truths = []
    constructed = []
    for i in used_indeces:
        closest_blocks = [cb[closest_codeblock_index(mat, block)] for block in train_blocks[i]]
        constructed_im = reconstruct_image(np.array(closest_blocks), train_im[i].shape)
        ground_truths.append(train_im[i])
        constructed.append(constructed_im)
    
    scores = self.ssim(ground_truths, constructed)
    return np.average(scores)