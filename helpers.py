# Helper functions
#
# 14 Feb 2014
# Goker Erdogan

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
    return np.argmax(y, axis=1)[:, np.newaxis]

def shuffle_dataset(x, y):
    """
    Shuffle dataset (x: input, y: labels)
    Returns shuffled x and y
    """
    rp = np.random.permutation(x.shape[0])
    x = x[rp,:]
    y = y[rp,:]
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

def save_dataset(dataset_name, path='./', prefix='', train_x=None, val_x=None, test_x=None, train_y=None, val_y=None, test_y=None):
    """
    Save training, validation and test data to disk.
    dataset_name: Dataset name (string)
    path: Path
    prefix: Filename prefix
    """
    if train_x is not None:
        np.save(path + prefix + dataset_name + '_train_x.npy', train_x)
    
    if val_x is not None:
        np.save(path + prefix + dataset_name + '_val_x.npy', val_x)
        
    if test_x is not None:
        np.save(path + prefix + dataset_name + '_test_x.npy', test_x)
    
    if train_y is not None:
        np.save(path + prefix + dataset_name + '_train_y.npy', train_y)
        
    if val_y is not None:
        np.save(path + prefix + dataset_name + '_val_y.npy', val_y)
        
    if test_y is not None:
        np.save(path + prefix + dataset_name + '_test_y.npy', test_y)
        
def load_dataset(dataset_name, path='./', prefix='', labeled=True, include_validation=True):
    """
    Load training, validation and test data from disk.
    dataset_name: Dataset name (string)
    path: Path
    prefix: Filename prefix
    labeled: If True, load labels
    include_validation: if True, load validation data
    Returns train_x, (validation_x), test_x, (train_y, (validation_y), test_y)
    """
    train_x = np.load(path + prefix + dataset_name + '_train_x.npy')
    test_x = np.load(path + prefix + dataset_name + '_test_x.npy')
    
    if not labeled:
        if include_validation:
            validation_x = np.load(path + prefix + dataset_name + '_val_x.npy')
            return train_x, validation_x, test_x
        else:
            return train_x, test_x
    else:
        train_y = np.load(path + prefix + dataset_name + '_train_y.npy')
        test_y = np.load(path + prefix + dataset_name + '_test_y.npy')
        if include_validation:
            validation_x = np.load(path + prefix + dataset_name + '_val_x.npy')
            validation_y = np.load(path + prefix + dataset_name + '_val_y.npy')
            return train_x, validation_x, test_x, train_y, validation_y, test_y
        else:
            return train_x, test_x, train_y, test_y
    