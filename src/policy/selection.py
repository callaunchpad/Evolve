import numpy as np
'''
fitness/selection/crossover/mutation.py
- abstract away the policies into respective files (for ease of experimentation)
- implement gradient mutation + single-point set crossover, 2x2x1 block size
- later: nested vector quantization, sepearate channel quantization
'''
def proportionate(fitness_vals):
    proportions = [val / sum(fitness_vals) for val in fitness_vals]
    indeces = [i for i in range(len(fitness_vals))]
    parents_to_keep = np.random.choice(indeces, len(indeces) // 2, proportions)

    offspring = []
    while len(offspring) < len(individuals) // 2:
        pts = np.random.randint(low=0, high=len(parents_to_keep), size=2)
        offspring.append((pts[0], pts[1]))
    return offspring