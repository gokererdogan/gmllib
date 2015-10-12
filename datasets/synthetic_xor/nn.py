# Multilayer feedforward neural network for xor dataset
# 14 Feb 2014
#
# Goker Erdogan

import gmllib.dataset as ds
import gmllib.neural_networks as nn
import gmllib.helpers as hlp

if __name__ == '__main__':
    # multilayer feedforward neural network example on synthetic xor data
    dataset = ds.DataSet(name='xor', trainx='xor_train_x.npy', trainy='xor_train_y.npy',
                         validationx='xor_val_x.npy', validationy='xor_val_y.npy',
                         testx='xor_test_x.npy', testy='xor_test_y.npy')
    # if you increase batch size, increasing momentum and epoch_count seems to work
    # epsilon does not seem to need adjusting. 0.1 is good.
    # H=2 also works
    w, b, val_pred, err = nn.nn_train(dataset, H=[4], epoch_count=10, epsilon=.1, batch_size=1, momentum=0.9)
    
    h = nn.nn_forward_pass(dataset.test.x.T, w, b)
    test_err = hlp.calculate_classification_error(dataset.test.y, h[-1].T)
    print('Test set classification error: ' + repr(test_err))
