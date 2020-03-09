def reconstruct(blocks, image_size):
    image = np.zeros(image_size)
    avg = np.zeros(image_size)
    bh = blocks.shape[1]
    bw = blocks.shape[2]
    for i in range(blocks.shape[0]):
        fitH = math.ceil(image_size[0]/bh)
        overH = image_size[0]%bh
        fitW = math.ceil(image_size[1]/bw)
        overW = image_size[1]%bw

        h0 = bh*(i//fitW)-overH if bh*(i//fitW)+bh>image_size[0] else bh*(i//fitW)
        h1 = h0+bh
        w0 = bw*(i%fitW)-overW if bw*(i%fitW)+bw>image_size[1] else bw*(i%fitW)
        w1 = w0+bw
        
        print('heights',h0,h1,'widths',w0,w1)
        avg[h0:h1,w0:w1] += np.ones((bh,bw))
        image[h0:h1,w0:w1] += blocks[i]
    return np.divide(image,avg)