import os
import numpy as np
import cv2
from policy import crossover, mutation, selection, fitness
import utils
from tqdm import tqdm

class GA():
    def __init__(self, pop_size, num_codeblocks,
        train_im, train_blocks, test_im, test_blocks,
        crossover_policy, fitness_policy, mutation_policy, selection_policy):
        
        self.pop_size = pop_size
        self.num_codeblocks = num_codeblocks
        self.block_size = train_blocks[0][0].shape
        
        self.fitness = fitness_policy
        self.crossover = crossover_policy
        self.mutation = mutation_policy
        self.selection = selection_policy

        self.train_im = train_im
        self.train_blocks = train_blocks
        self.test_im = test_im
        self.test_blocks = test_blocks

        self.population = []
        for i in range(pop_size):
            cb_set = []
            for j in range(num_codeblocks):
                cb_set.append(np.ones(self.block_size) * np.random.randint(0, 255))
            self.population.append(np.array(cb_set))
        self.population = np.array(self.population)
    
    def test_image_reconstruct(self, cb, image, image_blocks):
        mat = np.reshape(cb,
            (cb.shape[0], cb.shape[1] * cb.shape[2] * cb.shape[3])).T
        closest_blocks = [cb[utils.closest_codeblock_index(mat, block)] for block in image_blocks]
        constructed_im = utils.reconstruct_image(np.array(closest_blocks), image.shape)
        return constructed_im

    def step(self):
        fit_vals = []
        for i in tqdm(range(len(self.population))):
            val = self.fitness(self.population[i], self.train_im, self.train_blocks, batch_size = 1)
            fit_vals.append(val)
        
        pairs = self.selection(fit_vals)

        offspring = []
        for p in pairs:
            p1 = self.population[p[0]]
            p2 = self.population[p[1]]
            offspring.extend(self.crossover(p1, p2))
        
        self.population = np.array(offspring)
        self.population = self.mutation(self.population)

        best_set = np.argmax(fit_vals)
        return np.mean(fit_vals), np.std(fit_vals), fit_vals[best_set], self.population[best_set]
    
    def run(self, num_epochs):
        for epoch in range(100):
            print('---Epoch %d---' % (epoch + 1))
            mean, std, best, best_ind = evolve.step()
            print('mean: %f std: %f max: %f' % (mean, std, best))
            cv2.imwrite('current_best.png', evolve.test_image_reconstruct(best_ind, images[0], blocks[0]))

if __name__ == '__main__':
    images, blocks = np.load('../data/images/test_ims.npy'), np.load('../data/blocks/test_blocks.npy')
    evolve = GA(20, 25, images[:1], blocks[:1], images[1:], blocks[1:],
        crossover.block_one_point,
        fitness.ssim_reconstruct,
        mutation.gradient,
        selection.proportionate)
    
    evolve.run(100)
