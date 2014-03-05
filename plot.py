# Plotting functions
# Goker Erdogan
# 28 Jan 2013

import numpy as np
import neural_networks as nn
import matplotlib.pyplot as plt

def plot_dbn_sample(ws_vh, ws_v, ws_h, x, y=None, img_size=None, ax=None, k_step=10, total_samples=10):
    if img_size is None:
        raise Exception('img_size parameter should be provided.')
    
    plt.ion()
    ax = ax if ax is not None else plt.gca()
    ax.imshow(np.reshape(nn.gnp.as_numpy_array(x), img_size), cmap='gray')
    plt.show()
    plt.pause(0.3)
    s = x
    for i in range(total_samples):
        s = nn.dbn_sample(ws_vh, ws_v, ws_h, x=s, y=y, k=k_step)
        ax.clear()
        ax.imshow(np.reshape(nn.gnp.as_numpy_array(s), img_size), cmap='gray')
        plt.draw()
        plt.pause(0.1)

def hinton_diagram(x, ax=None):
    """
    Plot Hinton diagram of matrix x on axes ax.
    Modified from http://matplotlib.org/examples/specialty_plots/hinton_demo.html
    """
    ax = ax if ax is not None else plt.gca()
    
    # normalize to -1,1
    wx = x / np.max((np.abs(x)))
        
    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(wx):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

if __name__ == '__main__':
##    x = np.random.randn(20,20)
##    hinton_diagram(x)
##    plt.show()
    f = plt.figure()
    plot_dbn_sample(ws_vh, ws_v, ws_h, x, img_size=(28,28))
