# Convert mnist binary files to numpy arrays
# mnist files are available at http://yann.lecun.com/exdb/mnist/
#
# Goker Erdogan
# 4 Feb 2014

import numpy as np
import struct # for converting binary data

def read_labels(fname, N):
    # read test labels
    f = open(fname, 'rb')
    
    magic_number = f.read(4)
    magic_number = struct.unpack('>i', magic_number)[0]
    assert magic_number == 2049

    n = f.read(4)
    n = struct.unpack('>i', n)[0]
    assert n == N

    labels = np.zeros((n, 1), dtype=int)
    for i in range(n):
        labels[i,0] = struct.unpack('>B', f.read(1))[0]
    
    f.close()
    return labels    

def read_images(fname, N, D):
    f = open(fname, 'rb')
    magic_number = f.read(4)
    magic_number = struct.unpack('>i', magic_number)[0]
    assert magic_number == 2051

    n = f.read(4)
    n = struct.unpack('>i', n)[0]
    assert n == N
    
    d = f.read(4)
    d = struct.unpack('>i', d)[0]
    assert d == D
    d = f.read(4)
    d = struct.unpack('>i', d)[0]
    assert d == D
    
    images = np.zeros((n, d, d), dtype=int)
    for i in range(n):
        images[i,:,:] = np.reshape(struct.unpack('>784B', f.read(d*d)), (d,d))
    
    f.close()
    return images
    
if __name__ == '__main__':
    
    # read and save test labels
    test_labels = read_labels('t10k-labels-idx1-ubyte', N=10000)
    np.save('test_y.npy', test_labels) 

        
    # read and save test images
    test_images = read_images('t10k-images-idx3-ubyte', N=10000, D=28)
    np.save('test_x.npy', test_images)
    
    # read and save training labels
    train_labels = read_labels('train-labels-idx1-ubyte', N=60000)
    np.save('train_y.npy', train_labels) 

        
    # read and save training images
    train_images = read_images('train-images-idx3-ubyte', N=60000, D=28)
    np.save('train_x.npy', train_images)
    
    
