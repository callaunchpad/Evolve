import numpy as np
'''
fitness/selection/crossover/mutation.py
- abstract away the policies into respective files (for ease of experimentation)
- implement gradient mutation + single-point set crossover, 2x2x1 block size
- later: nested vector quantization, sepearate channel quantization
'''
def proportionate(fitness_vals, individuals, crossover, mutate):
    proportions = [val / sum(fitness_vals) for val in fitness_vals]
    parents_to_keep = np.random.choice(individuals, len(individuals) // 2, proportions)

    offspring = []
    while len(offspring) < len(individuals) // 2:
        pts = np.random.randint(low=0, high=len(parents_to_keep), size=2)
        child1, child2 = crossover(parents_to_keep[pts[0]], parents_to_keep[pts[1]])
        offspring.append(mutate(child1))
        offspring.append(mutate(child2))

    return np.hstack([parents_to_keep, offspring])