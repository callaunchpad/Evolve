import numpy as np
'''
fitness/selection/crossover/mutation.py
- abstract away the policies into respective files (for ease of experimentation)
- implement gradient mutation + single-point set crossover, 2x2x1 block size
- later: nested vector quantization, sepearate channel quantization
'''
def set_uniform(cb_1, cb_2):
    uniform = []
    uniform1 = []
    for i in range(cb_1.shape[0]):
        pt = np.random.randint(low=0, high=2, size=1)[0]
        if pt:
            uniform.append(cb_1[i])
            uniform1.append(cb_2[i])
        else:
            uniform.append(cb_2[i])
            uniform1.append(cb_1[i])
    return np.array(uniform), np.array(uniform1)

def set_one_point(cb_1, cb_2):
    pt = np.random.randint(low=0, high = cb_1.shape[0], size=1)[0]
    one_point = np.vstack((cb_1[:pt], cb_2[pt:]))
    one_point_1 = np.vstack((cb_2[:pt], cb_1[pt:]))
    return one_point, one_point_1

def set_two_point(cb_1, cb_2):
    pt = np.random.randint(low=0, high = cb_1.shape[0], size=2)
    two_point = np.vstack((cb_1[:pt[0]], cb_2[pt[0]:pt[1]], cb_1[pt[1]:]))
    two_point_1 = np.vstack((cb_2[:pt[0]], cb_1[pt[0]:pt[1]], cb_2[pt[1]:]))
    return two_point, two_point_1

def block_one_point(cb_1, cb_2):
    pts = np.round(np.random.normal(loc = 0.5, scale = 0.1, size = (cb_1.shape[0])) * cb_1.shape[1]).astype(np.int64)
    one_point_1 = np.array([np.vstack((cb_1[i][:pts[i]], cb_2[i][pts[i]:])) for i in range(cb_1.shape[0])])
    one_point_2 = np.array([np.vstack((cb_2[i][:pts[i]], cb_1[i][pts[i]:])) for i in range(cb_1.shape[0])])
    return one_point_1, one_point_2