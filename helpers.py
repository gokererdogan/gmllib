# Helper functions
#
# 14 Feb 2014
# Goker Erdogan

import math
import sys

import numpy as np

def convert_to_1ofK(y):
    """
    Convert labels 0-K to 1-of-K coding.
    Returns a NxK matrix
    """
    N = y.shape[0]
    K = np.max(y) + 1
    ry = np.zeros((N, K))
    for i in range(N):
        ry[i, y[i].astype(int)] = 1
    
    return ry


def convert_1ofK_to_ordinal(y):
    """
    Convert 1-of-K coding to ordinals (class labels from 0 to K)
    Returns a Nx1 vector
    """
    return np.argmax(y, axis=1)

def shuffle_dataset(x, y):
    """
    Shuffle dataset (x: input, y: labels)
    Returns shuffled x and y
    """
    rp = np.random.permutation(x.shape[0])
    x = x[rp]
    y = y[rp]
    return x, y

def normalize_dataset(x):
    """
    Normalize dataset such that minimum is 0 and maximum is 1.
    x: NxK matrix of input data
    Returns normalized data matrix 
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def calculate_classification_error(y, pred_y):
    """
    Calculate classification error
    y: True class labels. NxK matrix. 1-of-K coded.
    pred_y: Predicted class probabilities. NxK matrix.
    Returns percentage error (1-accuracy)
    """
    pred_labels = convert_1ofK_to_ordinal(pred_y)
    y_labels = convert_1ofK_to_ordinal(y)
    wrong = np.sum(pred_labels != y_labels)
    return wrong / float(y.shape[0])


def rgb2gray(rgb):
    """
    Convert RGB image to grayscale. Uses the same
    parameters with MATLAB's rgb2gray
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def progress_bar(current, max, label="", width=40, update_freq=1):
    """
    Print progress bar
        update_freq: print the progress bar every update_freq iteration. Useful if you want to update the progress bar
            more slowly
    """
    if current != 1 and (current % update_freq != 0):
        return
    percent = (current / float(max) * 100)
    filled = int(width * percent / 100.0)
    unfilled = width - filled
    done_str = ""
    if current >= max:
        done_str = "DONE\n"
    s = "\r[%{0:3d}][{1:s}>{2:s}] {3:s}{4:s}".format(int(percent), "=" * (filled-1), "-" * unfilled, label, done_str)
    sys.stdout.write(s)
    sys.stdout.flush()




    
