# Neural networks - Deep Learning
# Goker Erdogan
# 04 Feb 2013

import os

# Run on GPU (requires gnumpy and npmat)
USE_GPU = False
if USE_GPU:
    os.environ['GNUMPY_USE_GPU'] = 'yes'
else:
    os.environ['GNUMPY_USE_GPU'] = 'no'
        
# Warn when converting gnumpy arrays to numpy
IMPLICIT_CONVERSION = True
if IMPLICIT_CONVERSION:
    os.environ['GNUMPY_IMPLICIT_CONVERSION'] = 'warn'
else:
    os.environ['GNUMPY_IMPLICIT_CONVERSION'] = 'refuse'
    
import gnumpy as gnp
import numpy as np
import helpers as hlp
import time

# A NOTE ON TERMINOLOGY
# We can talk of two types of DBNs for our purposes: supervised and unsupervised.
# Although, in fact both of these are trained in an unsupervised manner from a
# theoretical point of view, we give them these two names because their structures
# are different.
# Unsupervised DBN: Only input to the network is from the bottom layer.
# Supervised DBN: Apart from the input at the bottom layer, the class labels are
# also given as input to the TOP LAYER RBM. (See Hinton, Osindero, Teh 2006 for
# an example of this for MNIST)

def dbn_sample_supervised(ws_vh, ws_v, ws_h, w, b, y, k=1):
    """
    WARNING: THIS IS SIMPLY WRONG!!! USE dbn_sample.
    Sample from Deep Belief Net that were trained supervised. Generate a sample
    with class y (where y uses 1-of-K coding)
    ws_vh, ws_v, ws_h: Lists of weights for Deep Belief Net
    w, b: Weight and bias matrix for the output layer (latent features to class labels)
    y: Sample class label (1-of-K coded)
    k: Number of Gibbs steps in RBM
    """
    # remember that the last layer is a usual single layer feedforward neural 
    # network attached to the whole dbn.
    # so we first need to go back from y (output) to features (activations of 
    # the last layer of dbn) 
    # however the outputs underdetermine the features (assuming that number of
    # features is greater than number of classes). so there are infinitely many
    # feature vectors that would give us the observed class label
    # what we do is we turn this into a fully determined problem by assigning
    # random values to extra features (there would be number_of_features - number_of_classes
    # such features) that cannot be determined. then we use a linalg.solve to
    # get feature values
    
    L = len(ws_vh) # number of layers
    K = y.shape[1] # number of classes
    H = w.shape[0] # number of latent features
    # we are solving w.T*x + b = y
    y = y.T - gnp.as_numpy_array(b)
    # assign random values to underdetermined outputs
    y = np.concatenate((y, np.random.rand(H-K, 1)))
    w_new = np.diag(np.ones(H))
    w_new[0:K,:] = gnp.as_numpy_array(w.T)
    x = np.linalg.solve(w_new, y)
    
    # the last layer of DBN is a RBM, we can use rbm_sample to sample from it
    # and propagate it down to the input layer
    # before we can call rbm_sample, we need to go from the output of rbm (which
    # we have) to the input.
    av = gnp.dot(ws_vh[-1], x) + ws_v[-1]
    v = gnp.logistic(av)
    
    # now we can call rbm_sample
    v = rbm_sample(ws_vh[-1], ws_v[-1], ws_h[-1], v, k)
    
    # propagate down to input
    # backward (top-down) pass
    for l in range(L-2,-1,-1):
        av = gnp.dot(ws_vh[l], v) + ws_v[l]
        v = gnp.logistic(av)
    
    return v
    
        
def rbm_sample(w_vh, w_v, w_h, x, k=1, clamped=None):
    """
    Sample from RBM with k steps of Gibbs sampling
    w_vh: Weights between visible and hidden units (matrix of size DxH)
    w_v: Visible unit biases (column vector of size Dx1)
    w_h: Hidden unit biases (column vector of size Hx1)
    x: Input (column vector of size DxN)
    k: Number of Gibbs steps. Default is 1.
    clamped: If not None, keeps the given elements of x clamped (constant)
        while sampling
        clamped is a two-tuple that gives the start and end indices of clamped elements
    Returns hidden unit and visible unit activations (matrices of size HxN, DxN)
    """
    if clamped is not None:
        cx = x[clamped[0]:clamped[1],:]
    
    v = x
    for i in range(k):
        # sample hiddens
        ah = gnp.dot(w_vh.T, v) + w_h
        h = gnp.logistic(ah)
        hs = (h > gnp.rand(h.shape[0], h.shape[1]))
        
        # sample visibles
        av = gnp.dot(w_vh, hs) + w_v
        v = gnp.logistic(av)
        
        if clamped is not None:
            v[clamped[0]:clamped[1],:] = cx
        
    return h, v

def rbm_train(train_x, H, batch_size, epoch_count, epsilon, momentum, return_hidden=True, verbose=True):
    """
    Train a (binary) restricted boltzmann machine.
    train_x: Input data. Matrix of size N (number of data points) x D (input dimension)
    H: Number of hidden units
    batch_size: Number of data points in each batch
    epoch_count: Number of training epochs
    epsilon: Learning rate, either a scalar or an array (one value for each epoch)
    momentum: Momentum parameter, either a scalar or an array (one value for each epoch)
    return_hidden: If True, returns hidden unit activations for training data. 
    verbose: If True, prints progress information
    Returns w_vh (weights between visible-hidden units), w_v (visible unit
    biases), w_h (hidden unit biases), h (hidden unit activations for input data),
    error (reconstruction error at each epoch)
    """
    N = train_x.shape[0]
    D = train_x.shape[1]
    batch_count = int(np.ceil(N / float(batch_size)))
    
    # if momentum is a scalar, create a list with the same value for all epochs
    if not isinstance(momentum, list):
        momentum = [momentum] * epoch_count
    if not isinstance(epsilon, list):
        epsilon = [epsilon] * epoch_count
    
    # initialize weights
    w_vh = gnp.randn((D, H)) * 0.1
    w_v = gnp.zeros((D, 1))
    w_h = gnp.zeros((H, 1))

    # weight updates
    dw_vh = gnp.zeros((D, H))
    dw_v = gnp.zeros((D, 1))
    dw_h = gnp.zeros((H, 1))
    
    # hidden unit activations
    if return_hidden:
        h = np.zeros((N, H)) # keep this a numpy array to save memory
    else:
        h = []

    start_time = time.time()
    # reconstruction errors over epochs
    error = []
    batch_order = range(batch_count)
    for e in range(epoch_count):
        if verbose:
            print('Epoch ' + repr(e+1))
        
        batch_error = []
        processed_batch = 0
        for b in range(batch_count):
            processed_batch += 1
            if verbose:
                print('\r%d/%d' % (processed_batch, batch_count)),
            
            start = b*batch_size
            end = (b+1)*batch_size if (b+1)*batch_size < N else N
            x = train_x[start:end, :].T
            
            # apply momentum
            dw_vh *= momentum[e]
            dw_v *= momentum[e]
            dw_h *= momentum[e]
            
            # positive phase
            ahp = gnp.dot(w_vh.T, x) + w_h
            hp = gnp.logistic(ahp)
            
            # if it is the last epoch, store hidden unit activations
            if return_hidden and e == epoch_count - 1:
                h[start:end, :] = gnp.as_numpy_array(hp.T)
            
            # add positive gradient term
            dw_vh += gnp.dot(x, hp.T)
            dw_v += gnp.sum(x, axis=1)[:,gnp.newaxis]
            dw_h += gnp.sum(hp, axis=1)[:,gnp.newaxis]
            
            # sample hiddens
            hs = (hp > gnp.rand(hp.shape[0], hp.shape[1]))

            # negative phase
            avn = gnp.dot(w_vh, hs) + w_v
            vn = gnp.logistic(avn)
            ahn = gnp.dot(w_vh.T, vn) + w_h
            hn = gnp.logistic(ahn)

            dw_vh -= gnp.dot(vn, hn.T)
            dw_v -= gnp.sum(vn, axis=1)[:,gnp.newaxis]
            dw_h -= gnp.sum(hn, axis=1)[:,gnp.newaxis]

            # update weights
            w_vh += epsilon[e]/(end-start) * dw_vh
            w_v += epsilon[e]/(end-start) * dw_v
            w_h += epsilon[e]/(end-start) * dw_h

            batch_error.append(gnp.mean((vn - x)**2))

        # shuffle batch order
        np.random.shuffle(batch_order)

        error.append(np.mean(batch_error))
        if verbose:
            print('\nReconstruction error: ' + repr(error[-1]))
            print('Elapsed time: ' + str(time.time() - start_time))
    
    return w_vh, w_v, w_h, h, error


def dbn_load(layer_count, path='./', file_prefix=''):
    """Temporary function for loading dbn weights from disk
    """
    ws_vh = []
    ws_v = []
    ws_h = []
    for i in range(layer_count):
        vh = np.load(path + file_prefix + 'L' + repr(i+1) + '_w_vh.npy')
        v = np.load(path + file_prefix + 'L' + repr(i+1) + '_w_v.npy')
        h = np.load(path + file_prefix + 'L' + repr(i+1) + '_w_h.npy')
        ws_vh.append(gnp.as_garray(vh))
        ws_v.append(gnp.as_garray(v))
        ws_h.append(gnp.as_garray(h))
    
    return ws_vh, ws_v, ws_h

def dbn_save(ws_vh, ws_v, ws_h, path='./', file_prefix=''):
    """Temporary function for saving dbn weights from disk
    """
    layer_count = len(ws_vh)
    for i in range(layer_count):
        np.save(path + file_prefix + 'L' + repr(i+1) + '_w_vh.npy', gnp.as_numpy_array(ws_vh[i]))
        np.save(path + file_prefix + 'L' + repr(i+1) + '_w_v.npy', gnp.as_numpy_array(ws_v[i]))
        np.save(path + file_prefix + 'L' + repr(i+1) + '_w_h.npy', gnp.as_numpy_array(ws_h[i]))
        
def dbn_forward_pass(ws_vh, ws_v, ws_h, x, y=None):
    """
    Deep belief net forward pass.
    x: input data (N x D matrix)
    y: Class label (1-of-K coded, N x K matrix). If not None, it is concatenated
        to the input for top layer RBM when calculating the output of the DBN.
    ws_vh: list of layer weights (L x D x H)
    ws_v: list of layer input biases (L x D x 1)
    ws_h: list of layer output biases (L x H x 1)
    Returns activations (continuous) and outputs (0-1, sigmoid(activations)) of
    top layer
    """
    L = len(ws_vh)
    h = x.T
    
    # forward (bottom-up) pass
    for l in range(L-1):
        ah = gnp.dot(ws_vh[l].T, h) + ws_h[l]
        h = gnp.logistic(ah)
    
    # if supervised, concatenate class labels to input to top layer RBM
    if y is not None:
        h = gnp.concatenate((y.T, h))
    
    ah = gnp.dot(ws_vh[-1].T, h) + ws_h[-1]
    h = gnp.logistic(ah)
    
    return ah.T, h.T

def dbn_supervised_predict_sample(ws_vh, ws_v, ws_h, x, k=20):
    """
    Predict the class label of input x from supervised DBN
    WARNING: THIS IS PRETTY SLOW AND LESS RELIABLE THAN THE EXACT METHOD
    Uses the sampling method mentioned in section 6.2 of Hinton, Osindero, Teh 2006
    x: Input data. (NxD matrix)
    k: Number of Gibbs steps
    """
    L = len(ws_vh)
    N = x.shape[0]
    
    # make a forward pass to get from input layer to visible layer of top level
    # RBM
    h_prev = x.T
    
    # forward (bottom-up) pass, (use deterministic (we pass the activations, not 
    # the stochastically sampled steps) forward pass)
    for l in range(L-1):
        ah = gnp.dot(ws_vh[l].T, h_prev) + ws_h[l]
        h_prev = gnp.logistic(ah)
    
    H = ws_vh[-1].shape[0] # number of visible units top level RBM
    Hx = h_prev.shape[0] # number of hidden units in the penultimate layer
    K = H - Hx
    # (H - Hx) is the number of supervised inputs to top level RBM
    # we give random values to the supervised portion of the input     
    v = gnp.concatenate((gnp.ones((K, N)) / K, h_prev))
    # we keep the visible units clamped while sampling
    h, v = rbm_sample(ws_vh[-1], ws_v[-1], ws_h[-1], v, k, clamped=(K, H))
    
    # sample visible units of top level RBM given 
    return v[0:K,:].T

def dbn_supervised_predict_exact(ws_vh, ws_v, ws_h, x):
    """
    Predict the class label of input x from supervised DBN
    Uses the exact method mentioned in section 6.2 of Hinton, Osindero, Teh 2006
    The free energy formula is taken from http://deeplearning.net/tutorial/rbm.html
    x: Input data. (NxD matrix)
    """
    L = len(ws_vh)
    N = x.shape[0]
    
    # make a forward pass to get from input layer to visible layer of top level
    # RBM
    h_prev = x.T
    
    # forward (bottom-up) pass, (use deterministic (we pass the activations, not 
    # the stochastically sampled steps) forward pass)
    for l in range(L-1):
        ah = gnp.dot(ws_vh[l].T, h_prev) + ws_h[l]
        h_prev = gnp.logistic(ah)
    
    H = ws_vh[-1].shape[0] # number of visible units top level RBM
    Hx = h_prev.shape[0] # number of hidden units in the penultimate layer
    K = H - Hx
    # (H - Hx) is the number of supervised inputs to top level RBM
    
    # for every class, assume it is the correct label and calculate its free energy
    y = gnp.zeros((K,N))
    free_energy = gnp.zeros((N,K)) # we actually calculate -free_energy
    for k in range(K):
        # set the current assumed class label
        y[k,:] = 1.0
        
        # visible unit vector
        v = gnp.concatenate((y, h_prev))
        e_v = gnp.dot(ws_v[-1].T, v) # bias energy term
        
        ah = gnp.dot(ws_vh[-1].T, v) + ws_h[-1]
        e_h = gnp.sum(gnp.log(gnp.exp(ah) + 1.0), axis=0)
        
        free_energy[:,k] = (e_v + e_h)
        
        # zero the class labels for next iteration
        y[:,:] = 0.0
    
    # since these numbers may get pretty small, use the sum-exp trick for converting
    # these to probabilities
    pred_y = gnp.exp(free_energy - gnp.max(free_energy, axis=1)[:,gnp.newaxis]) / gnp.sum(gnp.exp(free_energy - gnp.max(free_energy, axis=1)[:,gnp.newaxis]), axis=1)[:,gnp.newaxis]
    
    return pred_y
    

def dbn_sample(ws_vh, ws_v, ws_h, x, y=None, k=1):
    """
    Sample from DBN
    ws_vh, ws_v, ws_h: Lists of layer weights for DBN
    x: Initial sample. This is the input to DBN. (1xD vector)
    y: Class label for the sample. This corresponds to sampling from class
        conditionals. (1-of-K coded, row vector) 
    k: Number of Gibbs steps
    Returns a sample from DBN (1xD vector)
    """
    L = len(ws_vh)
    
    # make a forward pass to get from input layer to visible layer of top level
    # RBM
    h_prev = x.T
    
    # forward (bottom-up) pass
    for l in range(L-1):
        ah = gnp.dot(ws_vh[l].T, h_prev) + ws_h[l]
        h_prev = gnp.logistic(ah)
        h_prev = (h_prev > gnp.rand(h_prev.shape[0], h_prev.shape[1]))
    
    # if not supervised, sample from top layer RBM without clamping any of its
    # inputs
    if y is None:
        # sample from top layer RBM
        h, v = rbm_sample(ws_vh[-1], ws_v[-1], ws_h[-1], h_prev, k)
    else:
        K = y.shape[1] # number of classes
        H = ws_vh[-1].shape[0]
        # generate a random input to top layer RBM with class label units clamped to y
        v = gnp.concatenate((y.T, h_prev))
        # sample from top layer RBM
        h, v = rbm_sample(ws_vh[-1], ws_v[-1], ws_h[-1], v, k, clamped=(0,K))
        v = v[K:H,:]
        
    # backward (top-down) pass
    # propagate sample from RBM back to input
    for l in range(L-2,-1,-1):
        av = gnp.dot(ws_vh[l], v) + ws_v[l]
        v = gnp.logistic(av)
        
    return v.T

def dbn_train(train_x, H, batch_size, epoch_count, epsilon, momentum, 
                train_y=None, return_hidden=True, verbose=True):
    """
    NOTE: SUPERVISED TRAINING IS NOT REALLY TESTED WELL. TEST IT SOMEDAY!!!
    Unsupervised layerwise training of a sigmoidal Deep Belief Net.
    train_x: Training data. NxD matrix.
    train_y: Training labels NxK matrix (1-of-K coded). If provided, labels
        are included in the inputs to top layer RBM (See Hinton, Osindero, Teh 2006)
    H: Number of hidden units in each layer. e.g. [100, 2000, 300]
    batch_size: Batch size. Either a scalar or a list (epoch count for each layer).
    epsilon: Learning rate. Either a scalar or a list (an epsilon for each layer and epoch).
    momentum: Momentum. Either a scalar or a list (an epsilon for each layer and epoch).
    return_hidden: If True, returns hidden unit activations for training data.
    verbose: If True, prints progress information
    Returns ws_vh (list of weight matrices for each layer), ws_v (list of input 
    unit biases for each layer), ws_h (list of output unit biases for each layer),
    and, if return_hidden is True, h (output layer hidden unit activations for training data)
    """
    layer_count = len(H)
    # if any of the training parameters are given as scalars, convert them to lists
    if not isinstance(epoch_count, list):
        epoch_count = [epoch_count] * layer_count
    if not isinstance(batch_size, list):
        batch_size = [batch_size] * layer_count
    if not isinstance(epsilon, list):
        epsilon = [[epsilon] * e_c for e_c in epoch_count]
    if not isinstance(momentum, list):
        momentum = [[momentum] * e_c for e_c in epoch_count]
        
    ws_vh = []
    ws_v = []
    ws_h = []
    error = []
    # train layer by layer
    h = train_x
    for i, h_count in enumerate(H):
        # we need to return the hidden unit activations only for output layer, if
        # return_hidden is True
        if not return_hidden and i == layer_count - 1:
            rh = False
        else:
            rh = True
        
        # if we have train_y and we are training the last layer, concatenate
        # class labels to inputs
        if train_y is not None and i == layer_count - 1:
            h = gnp.concatenate((train_y, h), axis=1)
            
        w_vh, w_v, w_h, h, l_error = rbm_train(h, h_count, batch_size[i], epoch_count[i], epsilon[i], 
                momentum[i], return_hidden=rh, verbose=verbose)

        ws_vh.append(w_vh)
        ws_v.append(w_v)
        ws_h.append(w_h)
        error.append(l_error)
        
    return ws_vh, ws_v, ws_h, h, error
    
def dbn_supervised_finetune(train_x, train_y, validation_x, validation_y, 
            w_vh, w_h, batch_size=1, epoch_count=10, epsilon=0.01, 
            momentum=0.0, stop_if_val_error_increase=False, verbose=True):
    """
    WARNING: THIS MAY NOT BE THE BEST WAY TO TUNE WEIGHTS DISCRIMINATIVELY. THIS
        JUST TRAINS THE NETWORK USING BACKPROP; IT MAY BE BETTER TO USE
        SOME KIND OF WAKE-SLEEP ALGORITHM GIVEN IN (Hinton, Osindero, Teh 2006)
    Fine-tune Deep Belief Net weights in a supervised manner.
    Adds an output layer (softmax) and uses backprop to fine tune weights
    (Note that w_v are not needed, since network is only used in forward manner)
    train_x: NxD matrix of training data
    train_y: NxK vector of training data labels. (Should be coded using 1ofK coding)
    validation: VnxD matrix of validation data
    validation_y: VnxK vector of validation data labels
    w_vh: List of weight matrices for each layer.
    w_h: Input unit biases. List of bias vectors for each layer 
    batch_size: Batch size
    epoch_count: Epoch count
    epsilon: Learning rate
    momentum: Momentum
    stop_if_val_error_increase: If True, training is stopped when error on 
        validation set increases
    verbose: If True, prints progress information.
    Returns weights (list of weight matrices for each layer), input biases (list of
    bias vectors for each layer), validation predicted class labels and validation
    errors for each epoch
    """
    K = train_y.shape[1]
    H = []
    layer_count = len(w_vh)
    for l in range(layer_count):
        H.append(w_vh[l].shape[1])
    
    # add random initial weights for the output layer to list of weight matrices
    init_w = w_vh + [gnp.randn((H[-1], K)) * 0.01]
    init_b = w_h + [gnp.randn((K, 1)) * 0.01]
        
    w, b, val_pred, err = nn_train(train_x, train_y, validation_x, validation_y,
                                    H, init_w=init_w, init_b=init_b, batch_size=batch_size,
                                    epoch_count=epoch_count, epsilon=epsilon,
                                    momentum=momentum, 
                                    stop_if_val_error_increase=stop_if_val_error_increase,
                                    verbose=verbose)
    
    return w, b, val_pred, err
    

def nn_load(layer_count, path='./', file_prefix=''):
    """Temporary function for loading neural network weights from disk
    """
    w = []
    b = []
    for i in range(layer_count):
        wi = np.load(path + file_prefix + 'L' + repr(i+1) + '_w.npy')
        bi = np.load(path + file_prefix + 'L' + repr(i+1) + '_b.npy')
        w.append(gnp.as_garray(wi))
        b.append(gnp.as_garray(bi))
    
    return w, b

def nn_save(w, b, path='./', file_prefix=''):
    """Temporary function for saving neural network weights to disk
    """
    layer_count = len(w)
    for i in range(layer_count):
        np.save(path + file_prefix + 'L' + repr(i+1) + '_w.npy', gnp.as_numpy_array(w[i]))
        np.save(path + file_prefix + 'L' + repr(i+1) + '_b.npy', gnp.as_numpy_array(b[i]))
    
    
def nn_train(train_x, train_y, validation_x, validation_y, H, 
            init_w=None, init_b=None, batch_size=1, 
            epoch_count=10, epsilon=0.01, momentum=0.0, 
            stop_if_val_error_increase=False, verbose=True):
    """
    Multilayer feed-forward sigmoid neural network training with backpropagation.
    Hidden units have sigmoid non-linearity. 
    Output is soft-max.
    train_x: NxD matrix of training data
    train_y: NxK vector of training data labels. (Should be coded using 1ofK coding)
    validation: VnxD matrix of validation data
    validation_y: VnxK vector of validation data labels
    init_w: Initial weights. List of weight matrices for each layer.
    init_b: Initial biases. List of bias vectors for each layer 
    H: number of hidden layers in each layer as a list. e.g., [100, 50]
    batch_size: Batch size
    epoch_count: Epoch count
    epsilon: Learning rate
    momentum: Momentum
    stop_if_val_error_increase: If True, training is stopped when error on 
        validation set increases
    verbose: If True, prints progress information.
    Returns weights (list of weight matrices for each layer), biases (list of
    bias vectors for each layer), validation predicted class labels and validation
    errors for each epoch
    """
    N = train_x.shape[0]
    D = train_x.shape[1]
    K = train_y.shape[1]
    VN = validation_x.shape[0]
    batch_count = int(np.ceil(N / float(batch_size)))
    
    # if momentum is a scalar, create a list with the same value for all epochs
    if not isinstance(momentum, list):
        momentum = [momentum] * epoch_count
    
    H = [D] + H + [K]
    layer_count = len(H) - 1 # do not count input layer
    # initialize weights
    w = []
    b = []
    if init_w is None or init_b is None:
        for l in range(layer_count):
            input_dim = H[l]
            output_dim = H[l+1]
            w.append(gnp.randn((input_dim, output_dim)) * 0.01)
            b.append(gnp.randn((output_dim, 1)) * 0.01)
    else:
        w = init_w
        b = init_b
    
    # weight updates
    dw = []
    db = []
    for l in range(layer_count):
        input_dim = H[l]
        output_dim = H[l+1]
        dw.append(gnp.zeros((input_dim, output_dim)))
        db.append(gnp.zeros((output_dim, 1)))    
    
    start_time = time.time()
    # validation error over epochs
    val_error = []
    batch_order = range(batch_count)
    for e in range(epoch_count):
        if verbose:
            print('Epoch ' + repr(e+1))
        
        processed_batch = 0
        for batch_no in batch_order:
            processed_batch += 1
            if verbose:
                print('\r%d/%d complete.' % (processed_batch, batch_count)),
            
            start = batch_no * batch_size
            end = (batch_no + 1) * batch_size if (batch_no + 1) * batch_size < N else N
            x = train_x[start:end, :].T
            t = train_y[start:end, :].T
            
            # ----------- forward pass ---------------------------------------
            h = nn_forward_pass(x, w, b) # h contains unit activations for each layer
            # ----------- forward pass END -----------------------------------
            
            # ---------- calculate error signals -----------------------------
            # backward pass
            d = [None] * layer_count # list of error signals for each layer
            
            # output layer
            de_output = h[-1] - t # last element of h is output of network
            d[layer_count-1] = de_output
            
            # all layers except the output layer
            for l in range(layer_count-2, -1, -1):
                d[l] = ((1 - h[l+1]) * h[l+1]) * gnp.dot(w[l+1], d[l+1])
            # -------- calculate error signals END --------------------------
            
            # --- calculate gradient (weight updates) and update weights -----
            for l in range(layer_count):
                # apply momentum
                dw[l] *= momentum[e]
                db[l] *= momentum[e]
                
                # calculate updated
                dw[l] += gnp.dot(h[l], d[l].T)
                db[l] += gnp.sum(d[l], axis=1)[:, gnp.newaxis]
                
                # update weights
                w[l] -= epsilon/(end-start) * dw[l]
                b[l] -= epsilon/(end-start) * db[l]
        
            # --- calculate gradient (weight updates) and update weights END---
            
        # calculate validation set error
        val_pred_y = nn_forward_pass(validation_x.T, w, b, return_all=False)
        #val_pred_y = val_pred[-1].T
        # calculating classification error takes time, instead we calculate
        # squared difference between outputs
        #e_err = hlp.calculate_classification_error(validation_y, val_pred_y)
        # TEMP HACK for handling both numpy and gnumpy arrays
        if isinstance(validation_y, gnp.garray):
            e_err = gnp.sum((val_pred_y.T - validation_y)**2) / VN
        else:
            e_err = np.sum((val_pred_y.T - validation_y)**2) / VN
        #e_err = 0
        #val_pred_y = []
        val_error.append(e_err)
        
        # shuffle batch order
        np.random.shuffle(batch_order)
        
        if verbose:
            print('\nClassification error on validation set: ' + repr(val_error[-1]))
            print('Elapsed time: ' + str(time.time() - start_time))
            
        # if validation error increases, stop training
        if e >0 and stop_if_val_error_increase:
            if val_error[-1] > val_error[-2]:
                break
    
    return w, b, val_pred_y, val_error

def nn_forward_pass(x, w, b, return_all=True):
    """
    Forward pass for multilayer feed-forward sigmoid neural network
    Hidden units have sigmoid non-linearity. 
    Output is soft-max.
    x: DxN matrix of input data
    w: Weights. List of weight matrices for each layer.
    b: Biases. List of bias vectors for each layer
    return_all: If True, returns hidden unit activations for each layer. If False
        just returns the output layer activations
    Returns a list h where each element is a matrix containing the activations
    for that layer. h[0] is input data x. 
    """
    # ---- TEMP HACK --------------
    # I should find a more seamless way of running in mixed (some operations
    # with numpy, some with gnumpy) mode.
    # I had to resort to this, because i needed the validation classification
    # step in nn_train to run on CPU with numpy. GPU ran out of memory.
    if isinstance(x, gnp.garray):
        use_gpu = True
    else:
        use_gpu = False
            
    layer_count = len(w)
    if return_all:
        hs = [x] # unit activations for each layer
    h = x
            
    # all layers except the output layer
    for l in range(layer_count-1):
        if use_gpu:
            a = gnp.dot(w[l].T, h) + b[l]
            h = gnp.logistic(a)
        else:
            a = np.dot(gnp.as_numpy_array(w[l]).T, h) + gnp.as_numpy_array(b[l])
            h = 1. / (1 + np.exp(-a))
        if return_all:
            hs.append(h)
    
    # output layer
    if use_gpu:
        h = gnp.dot(w[-1].T, h) + b[-1]
        h = gnp.exp(h) / gnp.sum(gnp.exp(h), axis=0) # soft-max
    else:
        h = np.dot(gnp.as_numpy_array(w[-1]).T, h) + gnp.as_numpy_array(b[-1])
        h = np.exp(h) / np.sum(np.exp(h), axis=0) # soft-max

    if return_all:
        hs.append(h)
        return hs
    else:
        return h

if __name__ == '__main__':
    pass
