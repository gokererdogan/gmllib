# Multilayer feedforward neural network for xor dataset
# 14 Feb 2014
#
# Goker Erdogan

import neural_networks as nn
import helpers as hlp

if __name__ == '__main__':
    # multilayer feedforward neural network example on synthetic xor data
    train_x, validation_x, test_x, train_y, validation_y, test_y = hlp.load_dataset('xor', './datasets/synthetic_xor/')
    train_vy = hlp.convert_to_1ofK(train_y)
    validation_vy = hlp.convert_to_1ofK(validation_y)
    test_vy = hlp.convert_to_1ofK(test_y)
    # if you increase batch size, increasing momentum and epoch_count seems to work
    # epsilon does not seem to need adjusting. 0.1 is good.
    # H=2 also works
    w, b, val_pred, err = nn.nn_train(train_x, train_vy, validation_x, 
        validation_vy, H=[4], epoch_count=10, epsilon=.1, batch_size=1, momentum=0.9)
    
    h = nn.nn_forward_pass(test_x.T, w, b)
    test_err = hlp.calculate_classification_error(test_vy, h[-1].T)
    print('Test set classification error: ' + repr(test_err))