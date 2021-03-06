import numpy as np
from scipy.ndimage import gaussian_filter
'''
fitness/selection/crossover/mutation.py
- abstract away the policies into respective files (for ease of experimentation)
- implement gradient mutation + single-point set crossover, 2x2x1 block size
- later: nested vector quantization, sepearate channel quantization
'''
def reroll(codeblocks, mutation_proportion=0.15):
    replace = (np.random.random(codeblocks.shape) < mutation_proportion) * 1
    new_codeblocks = (codeblocks * (1 - replace) + np.random.randint(0, 256, codeblocks.shape) * replace)
    return new_codeblocks

def reroll_gaussian(codeblocks):
    re_rolled = reroll(codeblocks)
    print(re_rolled.shape)
    if (len(re_rolled.shape) == 4):
        g_rerolled = gaussian_filter(re_rolled, sigma=(0, 0, 1, 1))
    else:
        g_rerolled = gaussian_filter(re_rolled, sigma=(0, 0, 1, 1, 0.1))
    return np.array(g_rerolled)

def gradient(codeblocks, pop_rate=0.25, block_rate=0.25):
    N, B, H, W, D = codeblocks.shape #N, B, 4, 4, 3
    sample_size = int(pop_rate*N)
    choices = np.random.choice(np.arange(N), size=sample_size, replace=False) #random indices to mutate
    for i in choices:
        block_sample_size = int(block_rate*B) #how many blocks
        blocks = np.random.choice(np.arange(B), size=block_sample_size, replace=False) #indices of blocks
        grad = np.random.normal(0, 25, size=(block_sample_size, H, W, D))
        codeblocks[i, blocks] = np.add(codeblocks[i, blocks], grad, casting='unsafe') 
        codeblocks[i, blocks] = np.clip(codeblocks[i, blocks], 0, 255)
        codeblocks[i, blocks] = codeblocks[i, blocks].astype(np.int64)
    return codeblocks