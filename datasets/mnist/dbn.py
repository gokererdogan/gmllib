# Deep Belief Net for MNIST
# Uses the parameters given by 
# Deep unsupervised learning on a desktop PC: a primer for cognitive scientists
#   Alberto Testolin, Ivilin Stoianov, Michele De Filippo De Grazia and Marco Zorzi
#
# Goker Erdogan
# 17 February 2014

import neural_networks as nn
import helpers as hlp

if __name__ == '__main__':
    # -------LOAD DATASET------------------------------------------------------
    train_x, validation_x, test_x, train_y, validation_y, test_y = hlp.load_dataset('mnist', './')
    
    train_x_gpu = nn.gnp.as_garray(train_x)
    test_x_gpu = nn.gnp.as_garray(test_x)
    val_x_gpu = nn.gnp.as_garray(validation_x)
    
    train_y_gpu = nn.gnp.garray(train_vy)
    test_y_gpu = nn.gnp.garray(test_vy)
    val_y_gpu = nn.gnp.garray(validation_vy)
    #--------LOAD DATASET END-------------------------------------------------#
    
##    # ------- TRAIN DBN (SUPERVISED) ----------------------------------------#
##    # train DBN (supervised) and save results
##    ws_vh, ws_v, ws_h, h, error = nn.dbn_train(train_x_gpu, train_y=train_y_gpu, 
##                H=[500,500, 2000], batch_size=125, 
##                epoch_count=50, epsilon=0.1, momentum=.95, return_hidden=False, verbose=True)
##
##    # save dbn weights            
##    nn.dbn_save(ws_vh, ws_v, ws_h, path='./datasets/mnist/', file_prefix='')
##    
##    # save learned features
##    j, train_h = nn.dbn_forward_pass(ws_vh, ws_v, ws_h, x=train_x_gpu, y=train_y_gpu)
##    j, val_h = nn.dbn_forward_pass(ws_vh, ws_v, ws_h, x=val_x_gpu, y=val_y_gpu)
##    j, test_h = nn.dbn_forward_pass(ws_vh, ws_v, ws_h, x=test_x_gpu, y=test_y_gpu)
##    hlp.save_dataset('mnist_dbn', './', train_x=train_h, val_x=val_h, test_x=test_h)
##
##    # code for loading the dbn back from disk
##    # ws_vh, ws_v, ws_h = nn.dbn_load(layer_count=3, path='./', file_prefix='')
##    # ------- TRAIN DBN (SUPERVISED) END-------------------------------------#
    
##    # ------- TRAIN DBN (UNSUPERVISED) ----------------------------------------#
##    # train DBN (unsupervised) and save results
##    ws_vh, ws_v, ws_h, h, error = nn.dbn_train(train_x_gpu, train_y=None, 
##                H=[500,500, 2000], batch_size=125, 
##                epoch_count=50, epsilon=0.1, momentum=.95, return_hidden=False, verbose=True)
##
##    # save dbn weights                
##    nn.dbn_save(ws_vh, ws_v, ws_h, path='./datasets/mnist/', file_prefix='unsup_')
##    
##    # save learned features
##    j, train_h = nn.dbn_forward_pass(ws_vh, ws_v, ws_h, x=train_x_gpu)
##    j, val_h = nn.dbn_forward_pass(ws_vh, ws_v, ws_h, x=val_x_gpu)
##    j, test_h = nn.dbn_forward_pass(ws_vh, ws_v, ws_h, x=test_x_gpu)
##    hlp.save_dataset('mnist_dbn_unsup', './', train_x=train_h, val_x=val_h, test_x=test_h)
##    
##    # code for loading the dbn back from disk
##    # ws_vh, ws_v, ws_h = nn.dbn_load(layer_count=3, path='./', file_prefix='unsup_')
##    # ------- TRAIN DBN (UNSUPERVISED) END-------------------------------------#
    
    # ---- TEST DBN (SUPERVISED) ---------------------------------------------#
    # use DBN to predict outputs for test set
    
    # load supervised dbn
    ws_vh, ws_v, ws_h = nn.dbn_load(layer_count=3, path='./', file_prefix='')
    
    test_pred_h = nn.dbn_supervised_predict_sample(ws_vh, ws_v, ws_h, test_x_gpu, k=20)
    test_err_sup_s = hlp.calculate_classification_error(test_y_gpu, test_pred_h)
    print('Test set classification error with supervised DBN (sample): ' + repr(test_err_sup_s))
    # error about 13%

    test_pred_h = nn.dbn_supervised_predict_exact(ws_vh, ws_v, ws_h, test_x_gpu)
    test_err_sup_e = hlp.calculate_classification_error(test_y_gpu, test_pred_h)
    print('Test set classification error with supervised DBN (exact): ' + repr(test_err_sup_e))
    # error about 5.5%
    # ---- TEST DBN (SUPERVISED) END------------------------------------------#
    
    
##    # ----TEST DBN (UNSUPERVISED) --------------------------------------------#
##    # train a single layer neural network on the learned latent features (output of DBN)
##    
##    # What we do and don't do here:
##    #   Now we have trained a DBN on MNIST, we have extracted latent features
##    #   these are the outputs of the last layer of DBN (2000 features).
##    #   we can train a single layer neural network on these features to predict
##    #   the class labels.
##    #   note that we use the UNSUPERVISED trained DBN here, because it does not
##    #   make sense to use the supervised one, since the learned features in the
##    #   supervised trained DBN already contain information about the class labels
##    #   note that for example when we want to get the learned features for test
##    #   set from unsupervised DBN we need to give the test set labels alongside
##    #   so it does not make sense to then use these features to predict class
##    #   label, because they will just extract the information that was available
##    #   in the input. If you trained a nn on the supervised DBN's features, you
##    #   would get a perfect classifier (I did this, the error on test set is 
##    #   unsurprisingly 0!)
##    
##    # read data (learned (unsupervised) features)
##    dbn_train_x, dbn_val_x, dbn_test_x = hlp.load_dataset('mnist_dbn_unsup', './', labeled=False)
##    dbn_train_x_gpu = nn.gnp.as_garray(dbn_train_x)
##    dbn_val_x_gpu = nn.gnp.as_garray(dbn_val_x)
##    dbn_test_x_gpu = nn.gnp.as_garray(dbn_test_x)
##    # train neural network
##    w_h, b_h, val_pred_h, val_err_h = nn.nn_train(dbn_train_x_gpu, train_y_gpu, 
##                                    dbn_val_x_gpu, val_y_gpu, H=[], 
##                                    batch_size=125, epoch_count=10, epsilon=0.1, 
##                                    momentum=0.9)
##    
##    test_pred_h = nn.nn_forward_pass(dbn_test_x_gpu.T, w_h, b_h, return_all=False)
##    test_err_h = hlp.calculate_classification_error(test_y_gpu, test_pred_h.T)
##    print('Test set classification error with single layer nn trained on learned features: ' + repr(test_err_h))
##    # error is about 1.7%
##        
##    # save the neural network weights
##    nn.nn_save(w_h, b_h, path='./', file_prefix='nn_')
##    
##    # code to save nn weights to disk
##    # w_h, b_h = nn.nn_load(layer_count=1, path='./', file_prefix='nn_')
##    # ----TEST DBN (UNSUPERVISED) --------------------------------------------#

    #------------SAMPLE FROM DBN----------------------------------------------#
    # we use supervised DBN to sample from class conditionals
    # we get more meaningful samples if we start from an actual input from the
    # dataset
    import plot as mplt
    # this draws samples from dbn and plots them one after another
    mplt.plot_dbn_sample(ws_vh, ws_v, ws_h, train_x[3:4,:], y=train_y[3:4,:], 
                        img_size=(28,28), total_samples=50, k_step=20)

    #------------SAMPLE FROM DBN END------------------------------------------#
