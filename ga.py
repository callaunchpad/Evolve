import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'
import numpy as np
import cv2
from keras_contrib.losses import DSSIMObjective
import tensorflow as tf
import keras
from keras import backend as K
from image_utils import *
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
tf.logging.set_verbosity(tf.logging.ERROR)
# Class for Genetic Algorithm
class GA():

    # Initialize genetic algorithm with follows parameters
    def __init__(self, pop_size, num_centroids, block_size, train_im, train_blocks, test_im, 
        test_blocks, fitness_func = 'SSIM', crossover_policy = 'uniform', mating_policy = 'oleksii',
        selection_policy = 'proportionate', mutation_proportion = 0.05, 
        mutation_policy = 'reroll'):

        self.num_centroids = num_centroids
        self.block_size = block_size

        self.train_im = train_im
        self.test_im = test_im
        self.train_blocks = train_blocks
        self.test_blocks = test_blocks
        
        self.pop_size = pop_size
        self.individuals = []

        self.mutation_policy = mutation_policy
        self.crossover_policy = crossover_policy
        self.selection_policy = selection_policy
        self.mating_policy = mating_policy
        
        self.mutation_proportion = mutation_proportion
        self.fitness_func = fitness_func
        self.crossover_policy = crossover_policy
        self.selection_policy = selection_policy
        self.dssim = DSSIMObjective(kernel_size=3).__call__
        self.ssim = lambda im1, im2: 1 - K.get_value(self.dssim(tf.cast(np.array(im1), tf.float32), tf.cast(np.array(im2), tf.float32)))
        self.mse = lambda im1, im2: ((im1 - im2) ** 2).flatten().mean()
        
        for i in range(pop_size):
            self.individuals.append(Individual(num_centroids, block_size))

        self.epoch = 0            
    
    # function that creates an offspring for two individuals based on crossover policy
    def crossover(self, ind1, ind2):
        if self.crossover_policy == 'uniform':
            uniform = []
            for i in range(ind1.num_centroids):
                pt = np.random.randint(low=0, high=2, size=1)[0]
                if pt:
                    uniform.append(ind1.centroids[i])
                else:
                    uniform.append(ind2.centroids[i])
            return Individual(centroids = uniform)
        elif self.crossover_policy == 'one_point':
            pt = np.random.randint(low=0, high=ind1.num_centroids, size=1)[0]
            one_point = np.vstack((ind1.centroids[:pt], ind2.centroids[pt:]))
            return Individual(centroids = one_point)
        elif self.crossover_policy == 'two_point':
            pt = np.random.randint(low=0, high=ind1.num_centroids, size=2)
            two_point = np.vstack((ind1.centroids[:pt[0]], ind2.centroids[pt[0]:pt[1]], ind1.centroids[pt[1]:]))
            return Individual(centroids = two_point)
        elif self.crossover_policy == 'centroid_one_point':
            pts = np.round(np.random.normal(loc = 0.5, scale = 0.1, size=(ind1.num_centroids))*ind1.centroids.shape[1]).astype(np.int64)
            one_point_1 = np.array([np.vstack((ind1.centroids[i][:pts[i]], ind2.centroids[i][pts[i]:])) for i in range(ind1.num_centroids)])
            one_point_2 = np.array([np.vstack((ind2.centroids[i][:pts[i]], ind1.centroids[i][pts[i]:])) for i in range(ind1.num_centroids)])
            return Individual(centroids = one_point_1), Individual(centroids = one_point_2)
        elif self.crossover_policy == 'centroid_freq':
            lc1 = list(ind1.centroids)
            lc2 = list(ind2.centroids)
            ind1_centroids = [x for _, x in sorted(zip(ind1.freq, lc1), key=lambda pair: pair[0], reverse=True)]
            ind2_centroids = [x for _, x in sorted(zip(ind2.freq, lc2), key=lambda pair: pair[0], reverse=True)]
            ind1_centroids = np.array(ind1_centroids[:len(lc1)//2])
            ind2_centroids = np.array(ind2_centroids[:len(lc2)//2])
            top_centroids = np.vstack((ind1_centroids, ind2_centroids))
            return Individual(centroids = top_centroids)
        
    # function that returns a fitnesss value for an individual based on a set of images
    def fitness(self, ind, batch_size = 2):
        #used_indeces = np.random.choice(self.train_im.shape[0], batch_size)
        used_indeces = [0]
        fitness_val_per_im = []
        

        mat = np.reshape(ind.centroids,
         (ind.centroids.shape[0], ind.centroids.shape[1] * ind.centroids.shape[2] * ind.centroids.shape[3])).T
        
        def closest_centroid(bl):
            bl_r = np.reshape(bl, (-1, 1))
            norm = np.linalg.norm(mat - bl_r, axis=0)
            return ind.centroids[np.argmin(norm, axis=0)]
        ground_truths = []
        constructed = []
        out = 0
        # for i in used_indeces:
        #     constructed_im = reconstruct(np.array([closest_centroid(block) for block in self.train_blocks[i]]), self.train_im[0].shape)
        #     cv2.imwrite('yo.png', constructed_im)
        #     out += self.mse(constructed_im, self.train_im[i])

        for i in used_indeces:
            # train_blocks[i] - (15200, 25, 25, 1). centroids - (64, 25, 25, 1)
            constructed_im = reconstruct(np.array([closest_centroid(block) for block in self.train_blocks[i]]), self.train_im[0].shape)
            ground_truths.append(self.train_im[i])
            constructed.append(constructed_im)
            cv2.imwrite('yo.png', constructed_im)
        scores = self.ssim(ground_truths, constructed)
        return np.average(scores)
        # return 1 / (out / len(used_indeces) / 3072)
        
    # function that mutates individual for more biological feel to algorithm
    def mutate(self, ind):
        if self.mutation_policy == 'reroll':
            c = ind.centroids
            replace = (np.random.random(c.shape)<self.mutation_proportion)*1
            newC = (c*(1-replace)+np.random.randint(0, 256, c.shape)*replace)
            return Individual(centroids = np.array(newC))
        if self.mutation_policy == 'reroll+gaussian':
            c = ind.centroids
            replace = (np.random.random(c.shape)<self.mutation_proportion)*1
            newC = (c*(1-replace)+np.random.randint(0, 256, c.shape)*replace)
            g_newC = gaussian_filter(newC, sigma=(0,1,1,0))
            return Individual(centroids = np.array(g_newC))

    # mates the current individuals by [i, i+1] and creates n offspring per couple
    def mate(self, mating_pool, n_offsprings):
        offspring = []
        if self.mating_policy == 'sequential':
            for i in [x for x in range(len(mating_pool)) if x % 2 == 0]:
                child = self.crossover(mating_pool[i], mating_pool[i + 1])
                for _ in range(n_offsprings // len(mating_pool)):
                    offspring.append(self.mutate(child))
        elif self.mating_policy == 'oleksii':
            while len(offspring) < n_offsprings:
                pts = np.random.randint(low=0, high=len(mating_pool), size=2)
                if self.crossover_policy == 'centroid_one_point':
                    child1, child2 = self.crossover(mating_pool[pts[0]], mating_pool[pts[1]])
                    offspring.append(self.mutate(child1))
                    offspring.append(self.mutate(child2))
                else:
                    child = self.crossover(mating_pool[pts[0]], mating_pool[pts[1]])
                    offspring.append(self.mutate(child))
        return offspring
    
    # runs one epoch of the mating process
    def iterate(self):
        print("-----Iteration%2d-----" % self.epoch)
        fitness_vals = [] 
        for i in tqdm(range(len(self.individuals))):
            fit = self.fitness(self.individuals[i])
            mult = 1 if fit > 0 else -1
            fitness_vals.append((fit ** 2) * mult)
        print(fitness_vals)
        print(self.train_im.shape)
        # Print fitness score
        temp = fitness_vals[:]
        temp.sort(reverse = True)
        print("Top 5 Fitness Average:", sum([((abs(val)**0.5) * (val/abs(val))) for val in temp[:5]]) / 5)
        
        if self.selection_policy == 'proportionate':
            proportions = [val / sum(fitness_vals) for val in fitness_vals]
            parents_to_keep = np.random.choice(self.individuals, len(self.individuals) // 2, proportions)
            new_offspring = self.mate(parents_to_keep, len(self.individuals) // 2)
            self.individuals = np.hstack([parents_to_keep, new_offspring])

        if self.epoch % 20 == 0:
            np.save("exper/curr_pop_"+str(self.epoch)+".npy", self.individuals, allow_pickle = True)
        self.epoch = self.epoch + 1
        #FIXME - add more selection policies
    
    def top_5_fitness(self):
        temp = self.individuals
        temp.sort(key = self.fitness, reverse = True)
        fitness_sum = 0
        for i in range(5):
            fitness_sum = fitness_sum + self.fitness(temp[i])
        return fitness_sum / 5

    # returns the optimal set of centroids
    def optimal_centroid(self):
        return max(self.individuals, key = self.fitness)

# An individual defined by a set centroids (blocks)
class Individual():
    def __init__(self, num_centroids = 100, block_size = (5, 5, 3), centroids = None):
        if centroids is not None:
            self.centroids = np.array(centroids)
            self.num_centroids = self.centroids.shape[0]
        else: 
            self.num_centroids = num_centroids
            self.centroids = []

            for i in range(num_centroids):
                #rand_im = np.random.randint(255, size=block_size,dtype=np.uint8)
                rand_num = np.random.randint(255)
                self.centroids.append(np.ones((block_size))*rand_num)

            self.centroids = np.array(self.centroids)
        self.freq = []
        for i in range(num_centroids):
            self.freq.append(0)
