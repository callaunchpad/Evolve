import numpy as np
'''
fitness/selection/crossover/mutation.py
- abstract away the policies into respective files (for ease of experimentation)
- implement gradient mutation + single-point set crossover, 2x2x1 block size
- later: nested vector quantization, sepearate channel quantization
'''
def proportionate(fitness_vals):
    proportions = fitness_vals / sum(fitness_vals)
    choices = np.random.choice(np.arange(len(fit_vals)), len(fitness_vals), p = proportions)
    pairs = [choices[i * 2 : (i + 1) * 2] for i in range((len(choices) + 1) // 2)]
    return pairs