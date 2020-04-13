import numpy as np
from scipy.ndimage import gaussian_filter
'''
fitness/selection/crossover/mutation.py
- abstract away the policies into respective files (for ease of experimentation)
- implement gradient mutation + single-point set crossover, 2x2x1 block size
- later: nested vector quantization, sepearate channel quantization
'''
def reroll(codeblocks):
    replace = (np.random.random(codeblocks.shape) < self.mutation_proportion) * 1
    new_codeblocks = (codeblocks * (1 - replace) + np.random.randint(0, 256, codeblocks.shape) * replace)
    return new_codeblocks

def reroll_gaussian(codeblocks):
    re_rolled = reroll(codeblocks)
    g_rerolled = gaussian_filter(re_rolled, sigma=(0, 1, 1, 0))
    return np.array(g_rerolled)