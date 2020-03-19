from ga import *
if __name__ == '__main__':
    ims = np.load('test_ims.npy')
    blocks = np.load('test_blocks.npy')
    train_im = ims[:150]
    train_blocks = blocks[:150]
    test_im = ims[150:]
    test_blocks = ims[150:]
    alg = GA(100, 10, (80, 80), train_im, train_blocks, test_im, test_blocks)
    for i in range(100):
        print(alg.top_5_fitness())
        alg.iterate()