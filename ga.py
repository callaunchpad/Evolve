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
# Class for Genetic Algorithm
class GA():

    # Initialize genetic algorithm with follows parameters
    def __init__(self, pop_size, num_centroids, block_size, train_im, train_blocks, test_im, 
        test_blocks, fitness_func = 'SSIM', crossover_policy = 'uniform', mating_policy = 'oleksii',
        selection_policy = 'proportionate', mutation_proportion = 0.2, 
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
        #self.mse = lambda im1, im2: K.get_value(K.mean(keras.losses.mean_squared_error(im1, im2)))
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
            one_point = np.hstack((ind1.centroids[:pt], ind2.centroids[pt:]))
            return Individual(centroids = one_point)
        elif self.crossover_policy == 'two_point':
            pt = np.random.randint(low=0, high=ind1.num_centroids, size=2)
            two_point = np.hstack((ind1.centroids[:pt[0]], ind2.centroids[pt[0]:pt[1]], ind1.centroids[pt[1]:]))
            return Individual(centroids = two_point)
        #FIXME - add more crossover policies
    
    # function that returns a fitnesss value for an individual based on a set of images
    def fitness(self, ind, batch_size = 2):
        #used_indeces = np.random.choice(self.train_im.shape[0], batch_size)
        used_indeces = [0, 100]
        fitness_val_per_im = []

        def closest_centroid(bl):
            close = self.mse(bl, ind.centroids[0])
            close_block = ind.centroids[0]
            for centroid in ind.centroids:
                error = self.mse(bl, centroid)
                if error < close:
                    close = error
                    close_block = centroid
            return centroid
        
        ground_truths = []
        constructed = []
        for i in used_indeces:
            constructed_im = reconstruct(np.array([closest_centroid(block) for block in self.train_blocks[i]]), self.train_im[0].shape)
            ground_truths.append(self.train_im[i])
            constructed.append(constructed_im)
        
        scores = self.ssim(ground_truths, constructed)
        return np.average(scores)
        
    # function that mutates individual for more biological feel to algorithm
    def mutate(self, ind):
        if self.mutation_policy == 'reroll':
            c = ind.centroids
            replace = (np.random.random(c.shape)<self.mutation_proportion)*1
            newC = (c*(1-replace)+np.random.randint(0, 256, c.shape)*replace)
            return Individual(centroids = np.array(newC))

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
                child = self.crossover(mating_pool[pts[0]], mating_pool[pts[1]])
                offspring.append(child)
        return offspring
    
    # runs one epoch of the mating process
    def iterate(self):
        print("-----Iteration%2d-----" % self.epoch)
        fitness_vals 
        for i in tqdm(range(len(self.individuals))):
            fitness_vals.append(self.fitness(self.individuals[i]) ** 2)
        
        if self.selection_policy == 'proportionate':
            proportions = [val / sum(fitness_vals) for val in fitness_vals]
            parents_to_keep = np.random.choice(self.individuals, len(self.individuals) * 3 // 4, proportions)
            new_offspring = self.mate(parents_to_keep, len(self.individuals) // 4)
            self.individuals = parents_to_keep + new_offspring

        if self.epoch % 20 == 0:
            np.save("exper/curr_pop_"+str(self.epoch)+".npy", self.individuals, allow_pickle = True)
        self.epoch = self.epoch + 1
        #FIXME - add more selection policies
    
    def top_5_fitness(self):
        temp = self.individuals
        temp.sort(key = self.fitness)
        fitness_sum = 0
        for i in range(5):
            fitness_sum = fitness_sum + self.fitness(temp[i])
        return fitness_sum / 5

    # returns the optimal set of centroids
    def optimal_centroid(self):
        return max(self.individuals, key = self.fitness)

# An individual defined by a set centroids (blocks)
class Individual():
    def __init__(self, num_centroids = 100, block_size = (5, 5), centroids = None):
        if centroids is not None:
            self.centroids = np.array(centroids)
            self.num_centroids = self.centroids.shape[0]
        else: 
            self.num_centroids = num_centroids
            self.centroids = []

            block_height, block_width = block_size
            for i in range(num_centroids):
                rand_im = np.random.randint(255, size=(block_height, block_width, 3),dtype=np.uint8)
                self.centroids.append(rand_im)

            self.centroids = np.array(self.centroids)
