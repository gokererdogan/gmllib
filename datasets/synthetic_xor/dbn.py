# Deep Belief Net for synthetic xor data
# This is just for testing the DBN implementation.
# Otherwise, it does not make sense to use DBN to model this dataset.
#
# Goker Erdogan
# 17 February 2014

import gmllib.dataset as ds
import gmllib.neural_networks as nn
import gmllib.helpers as hlp

if __name__ == '__main__':
    dataset = ds.DataSet(name='xor', trainx='xor_train_x.npy', trainy='xor_train_y.npy',
                         validationx='xor_val_x.npy', validationy='xor_val_y.npy',
                         testx='xor_test_x.npy', testy='xor_test_y.npy')

    """
    train_x_gpu = nn.gnp.garray(hlp.normalize_dataset(train_x))
    test_x_gpu = nn.gnp.garray(hlp.normalize_dataset(test_x))
    val_x_gpu = nn.gnp.garray(hlp.normalize_dataset(validation_x))
    
    train_y_gpu = nn.gnp.garray(train_vy)
    test_y_gpu = nn.gnp.garray(test_vy)
    val_y_gpu = nn.gnp.garray(validation_vy)
    """
    
    ws_vh, ws_v, ws_h, h, error = nn.dbn_train(dataset.train.x, H=[10,10], batch_size=100,
                epoch_count=50, epsilon=0.1, momentum=.95, return_hidden=False, verbose=True)
                
    # nn.dbn_save(ws_vh, ws_v, ws_h, path='./datasets/synthetic_xor/', file_prefix='unsupervised_')
    
    w, b, val_pred, err = nn.dbn_supervised_finetune(ws_vh, ws_h, dataset=dataset, batch_size=1, epoch_count=10, epsilon=0.1,
            momentum=0.9, stop_if_val_error_increase=False, verbose=True)
    
    h = nn.nn_forward_pass(dataset.test.x.T, w, b)
    test_err = hlp.calculate_classification_error(dataset.test.y, h[-1].T)
    print('Test set classification error: ' + repr(test_err))
    