# Deep Belief Net for synthetic xor data
# This is just for testing the DBN implementation.
# Otherwise, it does not make sense to use DBN to model this dataset.
#
# Goker Erdogan
# 17 February 2014

import neural_networks as nn
import helpers as hlp

if __name__ == '__main__':
    train_x, validation_x, test_x, train_y, validation_y, test_y = hlp.load_dataset('xor', './datasets/synthetic_xor/')
    train_vy = hlp.convert_to_1ofK(train_y)
    validation_vy = hlp.convert_to_1ofK(validation_y)
    test_vy = hlp.convert_to_1ofK(test_y)
    
    train_x_gpu = nn.gnp.garray(hlp.normalize_dataset(train_x))
    test_x_gpu = nn.gnp.garray(hlp.normalize_dataset(test_x))
    val_x_gpu = nn.gnp.garray(hlp.normalize_dataset(validation_x))
    
    train_y_gpu = nn.gnp.garray(train_vy)
    test_y_gpu = nn.gnp.garray(test_vy)
    val_y_gpu = nn.gnp.garray(validation_vy)
    
    ws_vh, ws_v, ws_h, h, error = nn.dbn_train(train_x_gpu, H=[10,10], batch_size=100, 
                epoch_count=50, epsilon=0.1, momentum=.95, return_hidden=False, verbose=True)
                
    nn.dbn_save(ws_vh, ws_v, ws_h, path='./datasets/synthetic_xor/', file_prefix='unsupervised_')
    
    w, b, val_pred, err = nn.dbn_supervised_finetune(train_x_gpu, train_y_gpu, val_x_gpu, 
            val_y_gpu, ws_vh, ws_h, batch_size=1, epoch_count=10, epsilon=0.1, 
            momentum=0.9, stop_if_val_error_increase=False, verbose=True)
    
    h = nn.nn_forward_pass(test_x_gpu.T, w, b)
    test_err = hlp.calculate_classification_error(test_y_gpu, h[-1].T)
    print('Test set classification error: ' + repr(test_err))
    