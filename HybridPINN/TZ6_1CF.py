## This python script contains the Hybrid PINN class and some custom scaling functions
## Version Description:
    ## Earliest fully developed prototype of the PINN that predicts one correction factor (MTR_cf).
    ## Biases are also added in this version
    ## Validation data input and predictions are also added.
    ## This version implements code to automatically configure the neural network weights, biases and other procesess based on the passed hl_dim (hidden layer dimensions) list.
## Author: Dilip Rajkumar

import pandas as pd
import numpy as np
from itertools import chain
from TZ6_FDDN_Lib import fddn_zeta_input, FDDN_Solver, FDDN_output_df_gen, zeta_values

# A random seed is used to reproduce the weight initialization values and training performance everytime this scipt is run.
np.random.seed(42)

MTR = 'R600_HD'
select_elements = ['MTR-600','R-610', 'R-611', 'R-612', 'R-613','CAOR-620_fwd', 'CAOR-620_mid', 'CAOR-620_aft', 'CAOR-621_fwd', 'CAOR-621_mid', 'CAOR-621_aft']

# Define custom MSE function
MSE = lambda y_hat,y_true:  np.mean((np.asarray(y_hat)-np.asarray(y_true))**2)

## Calculate Loss Coefficients
def SHR_Zeta_3D(n_holes,hole_dia,MTR_DuctArea,cf):
    '''
    Computes the Zeta with 3D Correction Factor (cf) for Single Hole Retrictors
    '''
    MTR_New_Area = n_holes * (np.pi/4) * (hole_dia / 1000)**2 # Divide dia by 1000 to convert mm to m
    f0_f1 = MTR_New_Area/MTR_DuctArea
    l_cross = 1/hole_dia
    zeta_dash = 0.13 + 0.34 * 10**(-(3.4 * l_cross + 88.4 * l_cross**2.3))
    zeta_SHR_1D = ((1 - f0_f1 + (zeta_dash**0.5) * (1 - f0_f1)**0.375)**2) * f0_f1**(-2) # 1D Zeta
    zeta_SHR_3D = zeta_SHR_1D * cf # Zeta with 3D Correction Factor
    return MTR_New_Area,zeta_SHR_3D

def NormbyMax(df,features):
    '''Normalizes each column(feature) in a dataframe by the max value in that column'''
    df_scaled = pd.DataFrame()
    features_max_val = {}
    for e in features:
        max_val = df[e].max()
        features_max_val[e] = max_val
        df_scaled[e] = df[e] / max_val
    return features_max_val,df_scaled[features]

def DeNormbyMax(df_scaled,max_val_dict,features):
    '''DeNormalizes each column(feature) in a dataframe by the max value in that column'''
    df = pd.DataFrame()
    for e in features:
        max_val = max_val_dict[e]
        df[e] = df_scaled[e] * max_val
    return df[features]

# Define Custom Class for Physics Informed Neural Network
class PINN(object):
    def __init__(self, MTR_epsi, learning_rate, activation_function_choice = 'lrelu', error_loss_choice = 'full'):
        self.MTR_epsi = MTR_epsi
        self.lr = learning_rate
        self.error_loss_choice = error_loss_choice
        self.cf_lambda_reg_factor = 400
        print('Activation Function selected:',activation_function_choice)
        print('Loss Function & Delta-k Used:',error_loss_choice)
        print('Fixed Learning_rate Used:',learning_rate)

        #### Define various activation functions
        def linear(x):
            return x

        def linearDerivative(x):
            return 1

        def relu(x):
            return x * (x > 0)

        def reluDerivative(z):
            return np.greater(z, 0).astype(float)

        def lrelu(x):
            y1 = ((x > 0) * x)
            y2 = ((x <= 0) * x * 0.01)
            return y1+y2

        def lreluDerivative(x, alpha=0.01): # https://stackoverflow.com/questions/48102882/how-to-implement-the-derivative-of-leaky-relu-in-python
            dx = np.ones_like(x)
            dx[x < 0] = alpha
            return dx

        #### Set self.activation_function to the choice of function
        if (activation_function_choice == 'relu') :
            self.activation_function = relu
            self.activation_function_derivative = reluDerivative

        if (activation_function_choice == 'lrelu') :
            self.activation_function = lrelu
            self.activation_function_derivative = lreluDerivative

        if (activation_function_choice == 'linear') :
            self.activation_function = linear
            self.activation_function_derivative = linearDerivative

    def initialize_parameters(self,input_nodes, hl_dim, output_nodes, X_train = None, seed_value = 42):
        np.random.seed(seed_value)
        # Initialize weights for the input layer to first hidden layer
        w_i_dict = { 'W1': np.random.normal(0.0, input_nodes**-0.5,(input_nodes, hl_dim[0]))}
        b_i_dict = { 'b1': np.ones((hl_dim[0], ))}
        # Initialize empty dictionary for the hidden layers
        w_h_dict = {}
        b_h_dict = {}

        # Initialize weights for the last hidden layer to output layer
        w_o_dict = {'W{}'.format(len(hl_dim)+1): np.random.normal(0.0, output_nodes**-0.5,(hl_dim[-1], output_nodes))}
        b_o_dict = { 'b{}'.format(len(hl_dim)+1): np.ones((output_nodes, ))}


        # Construct dictionary for the hidden layers weight matrix
        for w_i,l_i in zip(range(2, len(hl_dim)+2), range(0, len(hl_dim)-1)):
            print('Initializing weights from first hidden layer to last hidden layer')
            w_h_dict["W{}".format(w_i)] = np.random.normal(0.0, hl_dim[l_i]**-0.5, (hl_dim[l_i],hl_dim[l_i + 1]))
            b_h_dict["b{}".format(w_i)] = np.ones((hl_dim[l_i + 1], ))


        # Package and return model parameters as a dictionary
        model = dict(chain(w_i_dict.items(), b_i_dict.items(), w_h_dict.items(),b_h_dict.items(),w_o_dict.items(), b_o_dict.items()))

        if X_train != None:
            # Making CFs == 1s at start of training each HoV in singleHoV Loop mode
            print('\n\nNr of Hidden layers:',len(hl_dim),';\tNr_Neurons in last hidden layer:', hl_dim[-1])
            MTR_cf, _, _ = self.forward_prop(model, X_train)
            delta_b = 1 - np.asarray(MTR_cf)
            print('PRE-INITIALIZATION MTR_cf:',MTR_cf, type(MTR_cf))
            print('PRE-INITIALIZATION delta_b:',delta_b, type(delta_b), delta_b.shape)
            b_o_dict = { 'b{}'.format(len(hl_dim)+1): np.ones((output_nodes, )) + delta_b }
            model = dict(chain(w_i_dict.items(), b_i_dict.items(), w_h_dict.items(),b_h_dict.items(),w_o_dict.items(), b_o_dict.items()))

        print('\nINITIAL WEIGHTS & BIASES:')
        for k in model.keys():
            print("{}:".format(k),list(model[k]),type(model[k]), "Shape:",model[k].shape)
        return model

    def train_1p_seq(self, network_dim, feature_names, train_data, train_data_dia, train_data_idx, MTR_DuctArea, target_df, zeta_col_names, V_max_org, epochs=50):
        ''' Train the network on input features of one data point at a time sequentially till it attains convergence

            Arguments
            ---------
            model:          NN network with initialized weights and biases
            feature_names:  list of input feature names as a list
            train_data:     2D array of input features, each row is one data record, each column is a feature
            train_data_dia: diameter array input for calculating Zeta using the `SHR_Zeta_3D` function
            train_data_idx: indices of the training data in the current batch
            MTR_DuctArea:   Duct Area for the Mixer Tapping Restrictor
            target_df:      Source dataframe containing required columns for FDDN Process
            zeta_col_names: Zeta names of all the elements
            V_max_org:      Max Value of the input Flow Rate column(s)
            epochs:         nr. of epochs to train the model
        '''
        # Initialize list to store loss history
        N_i, hidden_layer_dim, output_nodes = network_dim
        MSE_flowloss_hist_train = []
        error_loss_hist_train = []
        losses_hist = {'HoV':[],'MSE_flowloss':[],'Error_loss':[]}
        cf_hist_train = {'HoV':[],'MTR_cf_hat':[], 'Early_Stopping_Reason':[],'FlowRate_Diff_(LTR-FDDN)':[]}

        for idx in train_data_idx:
            X_train = train_data[feature_names].loc[[idx]].values
            dia = np.atleast_1d(train_data_dia.loc[idx])
            model = self.initialize_parameters(N_i, hidden_layer_dim, output_nodes) # Add X_train to inputs for custom_bias_initialization
            HoV = target_df[['HoV']].loc[idx].values.tolist()
            print('\nBEGIN Neural Network training for {}'.format(HoV))
            for i in range(1,epochs):
                MTR_cf, MTR_cf_epsi, a_cache = self.forward_prop(model, X_train)
                V_true_LTR, V_hat_FDDN, V_hat_MTR_epsi_FDDN = self.FDDN_SolverProcess(MTR_cf, MTR_cf_epsi, idx, dia, MTR_DuctArea, zeta_col_names, target_df)
                # Backpropagation
                error, weight_gradients, bias_gradients = self.backward_prop(model, a_cache, X_train, np.asarray(V_true_LTR), np.asarray(V_hat_FDDN), np.asarray(V_hat_MTR_epsi_FDDN), np.asarray(MTR_cf), V_max_org)
                # Gradient descent parameter update - Assign new parameters to the model
                model = self.update_parameters(model, weight_gradients, bias_gradients)
                ## Check error threshold for early stopping
                if ((error) < 0.001): #0.02 for full error#  1e-5 for CF-Only #0.001 for Simplified Error
                    break_flag = 1
                    print('Error is very LOW (< 0.001):',(error))
                else:
                    break_flag = 0
                # Store Training history performance
                MSE_flowloss_hist_train.append(MSE(V_hat_FDDN, V_true_LTR))
                error_loss_hist_train.append(error)
                early_stopping_bf = self.early_stopping(error_loss_hist_train, 10) # Call early stopping function to monitor last 10 epochs of the loss array
                # Print loss
                print('EPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",round(MSE_flowloss_hist_train[i-1],4), ";\tError_loss (Train): ", round(error_loss_hist_train[i-1],4))
                if (break_flag == 1) or (early_stopping_bf == 1) or (i == (epochs - 1)):
                    print('EARLY STOPPING ACTIVATED - Terminating Neural Network Training for {}'.format(HoV[0]))
                    print('MSEFlow Loss History for {}:'.format(HoV[0]),MSE_flowloss_hist_train)
                    print('Error Loss History for {}:'.format(HoV[0]),error_loss_hist_train)
                    losses_hist['HoV'].append(HoV[0])
                    losses_hist['MSE_flowloss'].append(MSE_flowloss_hist_train)
                    losses_hist['Error_loss'].append(error_loss_hist_train)
                    cf_hist_train['HoV'].append(HoV[0])
                    cf_hist_train['MTR_cf_hat'].append(MTR_cf[0])
                    cf_hist_train['FlowRate_Diff_(LTR-FDDN)'].append(V_true_LTR[0] - V_hat_FDDN[0] )
                    if (break_flag == 1):
                        cf_hist_train['Early_Stopping_Reason'].append('ErrorConverged')
                    elif (early_stopping_bf == 1):
                        cf_hist_train['Early_Stopping_Reason'].append('ErrorLossStagned')
                    elif (i == (epochs - 1)):
                        cf_hist_train['Early_Stopping_Reason'].append('SetEpochsNrReached')
                    elif (idx == train_data_idx[-1]):
                        print('TRAINING PROCESS COMPLETED - All HoVs Converged')
                    MSE_flowloss_hist_train = []
                    error_loss_hist_train = []
                    break

        return MSE_flowloss_hist_train, error_loss_hist_train, cf_hist_train, losses_hist

    def train(self, model, feature_names, train_data, train_data_dia, train_data_idx, train_batch_size, MTR_DuctArea, target_df, zeta_col_names, V_max_org, val_data = None, val_data_dia = None, val_data_idx = None, val_batch_size = 3, epochs=200):
        ''' Train the network on input features of on a batch of points using minibatch gradient descent

            Arguments
            ---------
            model:          NN network with initialized weights and biases
            feature_names:  list of input feature names as a list
            train_data:     2D array of input features, each row is one data record, each column is a feature
            train_data_dia: diameter array input for calculating Zeta using the `SHR_Zeta_3D` function
            train_data_idx: indices of the training data in the current batch
            train_batch_size: no. of data points in the training batch
            MTR_DuctArea:   Duct Area for the Mixer Tapping Restrictor
            target_df:      Source dataframe containing required columns for FDDN Process
            zeta_col_names: Zeta names of all the elements
            V_max_org:      Max Value of the input Flow Rate column(s)
            val_data:       validation data points for making predictions at the end of every epochs
            val_data_dia:   diameter array for validation points
            val_data_idx:   indices of validation data in the current batch
            val_batch_size: no. of data points in the validation batch
            epochs:         nr. of epochs to train the model
        '''
        # Initialize list to store loss history
        MSE_flowloss_hist_train = []
        error_loss_hist_train = []
        MSE_flowloss_hist_val = []
        error_loss_hist_val = []
        flow_delta_val_mean = []
        cf_hist_train = {'HoV':[],'MTR_cf_hat':[],'FlowRate_Diff_(LTR-FDDN)':[],'ConvergedEpoch':[]}
        cf_hist_val = {'HoV':[],'MTR_cf_hat_val':[],'FlowRate_Delta_(FDDN - LTR)':[],'ConvergedEpoch':[]}
        weight_keys = [k for k, v in model.items() if k.startswith('W')]
        zero_idx_loc = train_data_idx.index(0)
        # Gradient descent - Loop over epochs
        for i in range(0, epochs):
            V_true_LTR_train_arr, V_hat_FDDN_train_arr, V_hat_MTR_epsi_FDDN_arr, break_flag_arr, cf_array  = [], [], [], [], []
            if (len(train_data_idx) == train_batch_size):
                batch_idx = train_data_idx
                print('\nUSING ENTIRE TRAIN DATASET in current TRAINING Batch:', batch_idx)
            else:
                batch_idx = np.random.choice(train_data_idx, replace = False, size = train_batch_size)
                print('\nIndicies Selected in Current TRAINING Batch:',batch_idx)
            X_train = train_data[feature_names].loc[batch_idx].values
            dia_train = train_data_dia.loc[batch_idx]
            activations_array = {}
            activations_array.update({'a{}'.format(a_i):[] for a_i in range(0,len(weight_keys)+1)})
            for X, row_id, dia in zip(X_train, batch_idx, dia_train):
                # Forward propagation
                MTR_cf, MTR_cf_epsi, a_cache = self.forward_prop(model, X)
                for k in activations_array.keys():
                    activations_array[k].append(a_cache[k])
                V_true_LTR, V_hat_FDDN, V_hat_MTR_epsi_FDDN = self.FDDN_SolverProcess(MTR_cf, MTR_cf_epsi, row_id, dia, MTR_DuctArea, zeta_col_names, target_df)
            ## Calculate Percentage Difference between flowrates for early stopping
                percentage_diff = 100 * 2 * (abs(V_true_LTR - V_hat_FDDN)) / (V_true_LTR + V_hat_FDDN)
                if (percentage_diff < 0.35): # % diff < 0.35%
                    break_flag = 1
                    print('% Diff between flowrates:',round(percentage_diff[0],3),'%')
                else:
                    break_flag = 0
                # Append elements into a list
                break_flag_arr.append(break_flag)
                V_true_LTR_train_arr.append(V_true_LTR[0])
                V_hat_FDDN_train_arr.append(V_hat_FDDN[0])
                V_hat_MTR_epsi_FDDN_arr.append(V_hat_MTR_epsi_FDDN[0])
                cf_array.append(MTR_cf)

            # Backpropagation
            error, weight_gradients, bias_gradients = self.backward_prop(model, activations_array, X_train, np.reshape(np.asarray(V_true_LTR_train_arr), (train_batch_size, 1)) ,np.reshape(np.asarray(V_hat_FDDN_train_arr), (train_batch_size, 1)), np.reshape(np.asarray(V_hat_MTR_epsi_FDDN_arr),(train_batch_size,1)), np.reshape(np.asarray(cf_array),(train_batch_size,1)), V_max_org)
            # Gradient descent parameter update - Assign new parameters to the model
            model = self.update_parameters(model, weight_gradients, bias_gradients)
            # Store Training history performance
            MSE_flowloss_hist_train.append(MSE(V_hat_FDDN_train_arr, V_true_LTR_train_arr))
            error_loss_hist_train.append(error)
            if (val_data_idx != None):
                ## Evaluate Validation dataset
                V_LTR_val_arr, V_FDDN_val_arr, cf_val_arr, flow_delta_val = [], [], [], []
                if (len(val_data_idx) == val_batch_size):
                    val_batch_idx = val_data_idx
                    print('\nUSING ENTIRE VALIDATION DATASET in current Batch:', val_batch_idx)
                else:
                    val_batch_idx = np.random.choice(val_data_idx, replace = False, size = val_batch_size)
                    print('\nIndicies Selected in VALIDATION Batch:',val_batch_idx)
                X_val = val_data[feature_names].loc[val_batch_idx].values
                dia_val = val_data_dia.loc[val_batch_idx]
                for X_input, row_idx, diameter in zip(X_val, val_batch_idx, dia_val):
                    # Forward propagation
                    MTR_cf_val, V_LTR, V_FDDN, delta_V = self.predict(model, X_input, row_idx, diameter, MTR_DuctArea, zeta_col_names, target_df)
                    hov_val = target_df[['HoV']].loc[row_idx].values.tolist()
                    percentage_diff_val = 100 * 2 * (abs(V_LTR - V_FDDN)) / (V_LTR + V_FDDN)
                    cf_val_arr.append(MTR_cf_val)
                    V_LTR_val_arr.append(V_LTR[0])
                    V_FDDN_val_arr.append(V_FDDN[0])
                    flow_delta_val.append(abs(delta_V))
                    if (percentage_diff_val < 0.35): # % diff < 0.35%
                        print('% Diff between flowrates (VALIDATION):',round(percentage_diff_val[0],3),'%')
                        cf_hist_val['HoV'].append(hov_val[0])
                        cf_hist_val['MTR_cf_hat_val'].append(MTR_cf_val)
                        cf_hist_val['FlowRate_Delta_(FDDN - LTR)'].append(delta_V)
                        cf_hist_val['ConvergedEpoch'].append(i)
                # Store validation data performance history
                val_error = self.cf_lambda_reg_factor * (1/(2 * val_batch_size)) * ( 1/V_max_org**2 * ((np.asarray(V_FDDN_val_arr) - np.asarray(V_LTR_val_arr))**2).sum() + (1/self.cf_lambda_reg_factor) * ((np.asarray(cf_val_arr) - 1)**2).sum() )  # Calculate val data error
                # val_error = (1/(2 * val_batch_size)) * ( 1/V_max_org**2 * ((np.asarray(V_FDDN_val_arr) - np.asarray(V_LTR_val_arr))**2).sum() + (1/self.cf_lambda_reg_factor) * ((np.asarray(cf_val_arr) - 1)**2).sum() )  # Calculate val data error
                error_loss_hist_val.append(val_error)
                MSE_flowloss_hist_val.append(MSE(V_FDDN_val_arr, V_LTR_val_arr))
                flow_delta_val_mean.append(np.mean(flow_delta_val))
            # Print loss
            print('EPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",round(MSE_flowloss_hist_train[i],4), ";\tError_loss (Train): ", round(error_loss_hist_train[i],4) , ";\tMSE Flowloss (Val): ", round(MSE_flowloss_hist_val[i],4), ";\tError_loss (Val): ",round(error_loss_hist_val[i],4))
            train_early_stopping_bf = self.early_stopping(error_loss_hist_train, 10) # Call early stopping function to monitor last 10 epochs of the train loss array
            val_early_stopping_bf = self.early_stopping(error_loss_hist_val, 10) # Call early stopping function to monitor last 10 epochs of the validation loss array
            if (1 in break_flag_arr): # Store converged HoVs
                row_ids = list(filter(lambda e: e != 0, list(batch_idx * np.asarray(break_flag_arr)))) #Multiply row_ID list with bf_list and remove 0s
                if(0 in list(batch_idx)) and (break_flag_arr[zero_idx_loc] == 1): # Manually include [0] index if it is in the current batch
                    row_ids.append(0)
                HoVs = target_df[['HoV']].loc[row_ids].values.tolist()
                print('Row_IDs Converged:',row_ids,'\t HoVs Converged:',HoVs)
                MTR_cfs =      list(filter(lambda e: e != 0, list(cf_array * np.asarray(break_flag_arr)))) #Multiply MTR_cf_array with bf_list and remove 0s
                V_LTR_final = list(filter(lambda e: e != 0, list(V_true_LTR_train_arr * np.asarray(break_flag_arr))))
                V_FDDN_final = list(filter(lambda e: e != 0, list(V_hat_FDDN_train_arr * np.asarray(break_flag_arr))))
                for hov,cf1,V_LTR,V_FDDN in zip(HoVs,MTR_cfs,V_LTR_final,V_FDDN_final):
                    cf_hist_train['HoV'].append(hov[0])
                    cf_hist_train['MTR_cf_hat'].append(cf1)
                    cf_hist_train['FlowRate_Diff_(LTR-FDDN)'].append(V_LTR - V_FDDN )
                    cf_hist_train['ConvergedEpoch'].append(i)
                # print('REMOVING CONVERGED HoVs:',HoVs,'Rows Dropped:',row_ids)
                # train_data_idx = list(set(train_data_idx).difference(set(row_ids)))
                # if (len(train_data_idx) < train_batch_size): # reduce batch size when available data is less than batch size
                #     train_batch_size = train_batch_size - len(row_ids)
            # if (len(train_data_idx) == 0):
            #     print('TRAINING PROCESS COMPLETED - All HoVs Converged')
            #     break
            if (train_early_stopping_bf == 1) and (val_early_stopping_bf == 1): # If both train and val error loss stagnates terminate neural network training
                break
            elif (i == epochs - 1):
                print('TRAINING PROCESS COMPLETED - Set Nr. of Training Epochs Reached')

        return model, MSE_flowloss_hist_train, error_loss_hist_train, cf_hist_train, MSE_flowloss_hist_val, error_loss_hist_val, cf_hist_val, flow_delta_val_mean

    def forward_prop(self, model, X):
        '''
        model - Model parameters (i.e weights and biases)
        X    - activations from the input layer (basically the input features denoted by 'X')
        '''
        # Load parameters from model
        a_dict = {}
        z_dict = {}

        weights = [k for k, v in model.items() if k.startswith('W')]

        # Create Empty dictionary with keys for the activations and linear combinations
        for i in range(0,len(weights)):
            a_dict.update({'a{}'.format(i):[]})
            z_dict.update({'z{}'.format(i+1):[]})

        # Assign first value in activations dict to features
        a_dict['a0'] = X

        for a_i, w_i in zip(range(0, len(weights)), range(1,len(weights)+1)):
            # print('a{}:'.format(a_i), list(a_dict['a{}'.format(a_i)]),'Shape:',a_dict['a{}'.format(a_i)].shape)
            # print('W{}:'.format(w_i), list(model['W{}'.format(w_i)]), 'Shape:',model['W{}'.format(w_i)].shape)
            # print('np.dot(a,W):', list( np.dot(a_dict['a{}'.format(a_i)], model['W{}'.format(w_i)]) ), 'Shape:', np.dot(a_dict['a{}'.format(a_i)], model['W{}'.format(w_i)]).shape)
            # print('b{}:'.format(w_i), list(model['b{}'.format(w_i)]),  'Shape:', model['b{}'.format(w_i)].shape)
            z_dict['z{}'.format(w_i)] = np.dot(a_dict['a{}'.format(a_i)], model['W{}'.format(w_i)])  + model['b{}'.format(w_i)] # adding biases
            # print('z{} Shape:'.format(w_i), list(z_dict['z{}'.format(w_i)]),  'Shape:',z_dict['z{}'.format(w_i)].shape)

            a_dict['a{}'.format(a_i + 1)] = self.activation_function(z_dict['z{}'.format(w_i)])
            # print('a{} Shape:'.format(a_i + 1), list(a_dict['a{}'.format(a_i + 1)]), 'Shape:', a_dict['a{}'.format(a_i + 1)].shape)

        # Assign last value of activations dict to last z, because we want to use linear activations in the output layer
        a_dict['a{}'.format(len(weights))] = z_dict[list(z_dict.keys())[-1]]

        c_f = a_dict[list(a_dict.keys())[-1]]
        # print('{}:'.format(list(a_dict.keys())[-1]),a_dict[list(a_dict.keys())[-1]])
        # Correction_factor output from final output layer having linear (identity) activation function
        MTR_cf_pred = c_f[0]

        if type(MTR_cf_pred) is np.float64:
            print('\nMTR-Cf:',MTR_cf_pred)
        elif type(MTR_cf_pred) is np.ndarray:
            print('\nMTR-Cf:',MTR_cf_pred[0])

        # print('Activations_cache:', list(a_dict.keys()) )
        if (MTR_cf_pred >= 0):
            MTR_cf_epsi = MTR_cf_pred + self.MTR_epsi
        elif (MTR_cf_pred < 0):
            MTR_cf_epsi = MTR_cf_pred - self.MTR_epsi
            MTR_cf_pred = abs(MTR_cf_pred)
            print('NN output is -ve. Taking |(MTR-cf)|:',MTR_cf_pred)

        return MTR_cf_pred, MTR_cf_epsi, a_dict

    def early_stopping(self,train_loss, last_n_epochs = 10):
        '''Monitors the last n epochs of the train loss and returns a break flag to stop the training process
            Arguments
            ---------
            train_loss: array of the train loss values stored at the end of every training epoch
            last_n_epochs: the last 'n' epochs of the train_loss array to monitor for change in magnitude. Default value = 10
        '''
        if len(train_loss) >= last_n_epochs:
            # print("Error Loss List:", train_loss, ",...List Length:",len(train_loss))
            list_elements_diff = np.diff(train_loss[-last_n_epochs:])
            loss_mag = np.linalg.norm(list_elements_diff)
        else:
            loss_mag = 1

        if (loss_mag < 0.01): #For CF error we use 0.001, for full error we use 0.01
            print('NO CHANGE in LOSS for the Last {} Epochs'.format(last_n_epochs))
            es_bf = 1
        else:
            es_bf = 0

        return es_bf

    def FDDN_SolverProcess(self, MTR_cf, MTR_cf_epsi, row_id, dia, MTR_DuctArea, zeta_col_names, target_df):
        ## Pass Neural network output (correction factor) to compute new Zeta and then re-run FDDN solver
        _, zeta_tuple = zip(*[SHR_Zeta_3D(1,dia,MTR_DuctArea,MTR_cf)])
        target_df[MTR+'_Zeta1D'].loc[row_id] = zeta_tuple[0] # Assign first element in zeta_tuple to required index
        # print('New_Zeta for MTR-600:' , target_df[MTR+'_Zeta1D'].loc[row_id])

        ## Call FDDN Solver to compute new flowrates with the updated Zeta
        mixp_input,ambp_input,ambt_input,zeta_input = fddn_zeta_input(target_df[['HoV','MIXP','AMBP','AMBT','TZ6_Flow']+zeta_col_names].loc[[row_id]])
        FDDN_FlowRates_raw_df = FDDN_Solver(select_elements,mixp_input,ambp_input,ambt_input,zeta_input)
        FDDN_FlowRates_df = FDDN_output_df_gen(FDDN_FlowRates_raw_df )
        V_hat_FDDN = FDDN_FlowRates_df['TZ6_Flow'].values

        if (MTR_cf_epsi != None): # For validation and test points we won't compute epsi as there is no Backpropagation
            ## Compute new flow rates for MTR-Epsi
            _, zeta_tuple_epsi = zip(*[SHR_Zeta_3D(1,dia, MTR_DuctArea, MTR_cf_epsi)])
            target_df[MTR+'_Zeta1D'].loc[row_id] = zeta_tuple_epsi[0] # Assign first element in zeta_tuple to required index
            mixp_input_epsi,ambp_input_epsi,ambt_input_epsi,zeta_input_epsi = fddn_zeta_input(target_df[['HoV','MIXP','AMBP','AMBT','TZ6_Flow']+zeta_col_names].loc[[row_id]])
            FDDN_FlowRates_raw_df_epsi = FDDN_Solver(select_elements,mixp_input_epsi,ambp_input_epsi,ambt_input_epsi,zeta_input_epsi)
            FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
            V_hat_MTR_epsi_FDDN = FDDN_FlowRates_df_epsi['TZ6_Flow'].values
        else:
            V_hat_MTR_epsi_FDDN = 0

        V_true_LTR = target_df['TZ6_Flow'].loc[[row_id]].values
        print('Row ID:',row_id,'\tHoV:', target_df['HoV'].loc[[row_id]].values)
        # print('LTR_TZ6_FlowRate:',V_true_LTR)
        # print('FDDN_TZ6_Flowrate:', V_hat_FDDN)
        print('FlowRate Difference (LTR - FDDN):',V_true_LTR[0] - V_hat_FDDN[0],'l/s')

        return V_true_LTR, V_hat_FDDN, V_hat_MTR_epsi_FDDN

    def backward_prop(self, model, activations, X, V_LTR, V_hat_FDDN, V_hat_MTR_epsi_FDDN, c_f, V_max):
        # Backpropagation
        m = X.shape[0] # Get number of samples in the current training batch

        # Load parameters from model
        weight_keys = [k for k, v in model.items() if k.startswith('W')]
        activations_keys = list(activations.keys())

        dz_dict = {}
        dW_dict = {}
        db_dict = {}
        for i in range(1,len(weight_keys)+1):
            dz_dict.update({'dz{}'.format(i):[]})
            dW_dict.update({'dW{}'.format(i):[]})
            db_dict.update({'db{}'.format(i):[]})

        dz_dict_keys = list(dz_dict.keys())
        dW_dict_keys = list(dW_dict.keys())
        db_dict_keys = list(db_dict.keys())

        # Derivative of flowrate with respect to neural network o/p (i.e correction factor)
        dv_da = (V_hat_MTR_epsi_FDDN - V_hat_FDDN) / self.MTR_epsi

        # Difference between the FDDN flowrate and LTR Flow Rate
        flowrate_diff = np.reshape(V_hat_FDDN.flatten('F') - V_LTR.flatten('F'), (m,1))
        # print('\nBACKWARD PROP')
        # print('Input Features:',list(X), type(X), X.shape)
        # print('V_max:',V_max, type(V_max))
        # print('cf_array:',list(c_f), type(c_f), c_f.shape)
        # print('dv_da:',list(dv_da), type(dv_da), dv_da.shape)
        # print('Flowrate_delta (FDDN - LTR):', list(flowrate_diff), type(flowrate_diff), flowrate_diff.shape)

        if (self.error_loss_choice == 'simplified'):
            error = self.cf_lambda_reg_factor * (1/(2 * m)) * ( 1/V_max**2 * (flowrate_diff**2).sum() )   # Calculate error
            dz_dict[dz_dict_keys[-1]] = self.cf_lambda_reg_factor * (1/m)  * (1/V_max**2) * (flowrate_diff * dv_da)
        elif (self.error_loss_choice == 'cf_only'):
            error = (1/(2 * m)) * ((c_f - 1)**2).sum() # Calculate error
            dz_dict[dz_dict_keys[-1]] =  (1/m) * (c_f - 1) # Delta-k - dE/dWjk
        elif (self.error_loss_choice == 'full'):
            error = self.cf_lambda_reg_factor * (1/(2 * m)) * ( 1/V_max**2 * (flowrate_diff**2).sum() + (1/self.cf_lambda_reg_factor) * ((c_f - 1)**2).sum() )  # Calculate error
            dz_dict[dz_dict_keys[-1]] = self.cf_lambda_reg_factor * (1/m) * ( 1/V_max**2 * flowrate_diff * dv_da  + (1/self.cf_lambda_reg_factor) *(c_f - 1) ) # Delta-k - dE/dWjk
            # error = (1/(2 * m)) * ( 1/V_max**2 * (flowrate_diff**2).sum() + (1/self.cf_lambda_reg_factor) * ((c_f - 1)**2).sum() )  # Calculate error
            # dz_dict[dz_dict_keys[-1]] = (1/m) * ( 1/V_max**2 * flowrate_diff * dv_da  + (1/self.cf_lambda_reg_factor) *(c_f - 1) ) # Delta-k - dE/dWjk
        elif (self.error_loss_choice == 'hyperbola_regularization'):
            error = self.cf_lambda_reg_factor * (1/(2 * m)) * ( 1/V_max**2 * (flowrate_diff**2).sum() + (1/self.cf_lambda_reg_factor) * (( max(c_f, 1 /c_f) - 1)**2).sum() )  # Calculate error
            dz_dict[dz_dict_keys[-1]] =  self.cf_lambda_reg_factor * (1/m) * ( 1/V_max**2 * flowrate_diff * dv_da  - (1/self.cf_lambda_reg_factor) * max((c_f - 1),((1 /c_f) - 1) * (1 /c_f)**2))  # Delta-k - dE/dWjk

        # print("{} - (Delta_k):".format(dz_dict_keys[-1]),list(dz_dict[dz_dict_keys[-1]]), "Shape:",dz_dict[dz_dict_keys[-1]].shape)
        dW_dict[dW_dict_keys[-1]] =  (1/m)*np.dot(np.asarray(activations[activations_keys[-2]]).T, dz_dict[dz_dict_keys[-1]])
        # print("{}:".format(dW_dict_keys[-1]),list(dW_dict[dW_dict_keys[-1]]), "Shape:",dW_dict[dW_dict_keys[-1]].shape)
        db_dict[db_dict_keys[-1]] = (1/m)*np.sum(dz_dict[dz_dict_keys[-1]], axis=0)
        # print("{}:".format(db_dict_keys[-1]),list(db_dict[db_dict_keys[-1]]), "Shape:",db_dict[db_dict_keys[-1]].shape)
        for w_i, k in zip(range(1,len(weight_keys)), range(2,len(activations)+1)):
            dz_dict[dz_dict_keys[-(k)]] = np.multiply(np.dot(dz_dict[dz_dict_keys[-(w_i)]], model[weight_keys[-(w_i)]].T), self.activation_function_derivative(np.asarray(activations[activations_keys[-(k)]])))
            # print("{}:".format(dz_dict_keys[-k]),list(dz_dict[dz_dict_keys[-k]]), "Shape:",dz_dict[dz_dict_keys[-k]].shape)
            dW_dict[dW_dict_keys[-(k)]] =  (1/m)*np.dot(np.asarray(activations[activations_keys[-(k+1)]]).T, dz_dict[dz_dict_keys[-(k)]])
            # print("{}:".format(dW_dict_keys[-k]),list(dW_dict[dW_dict_keys[-k]]), "Shape:",dW_dict[dW_dict_keys[-k]].shape)
            db_dict[db_dict_keys[-(k)]] =  (1/m)* np.sum(dz_dict[dz_dict_keys[-(k)]], axis=0)
            # print("{}:".format(db_dict_keys[-k]),list(db_dict[db_dict_keys[-k]]), "Shape:",db_dict[db_dict_keys[-k]].shape)
        return error, dW_dict, db_dict

    def update_parameters(self, model, weight_gradients, bias_gradients):
        # Load model parameters
        weights_keys = [k for k, v in model.items() if k.startswith('W')]
        biases_keys = [k for k, v in model.items() if k.startswith('b')]
        weight_gradients_keys = list(weight_gradients.keys())
        bias_gradients_keys = list(bias_gradients.keys())
        # print('UPDATE PARAMETERS - Weight_keys:', weights_keys )
        # print('UPDATE PARAMETERS - Biases_keys:', biases_keys )
        # print('UPDATE PARAMETERS - weight_gradients_keys:', weight_gradients_keys )
        # print('UPDATE PARAMETERS - bias_gradients_keys:', bias_gradients_keys )

        # Update parameters
        for w_k, dw_k, b_k, db_k in zip(weights_keys, weight_gradients_keys, biases_keys, bias_gradients_keys ):
            model[w_k] -= self.lr * weight_gradients[dw_k]
            model[b_k] -= self.lr * bias_gradients[db_k]
        # print('MODEL WEIGHTS & BIASES UPDATED')
        return model

    def predict(self, model, x, row_id, dia, MTR_DuctArea, zeta_col_names, target_df):
        # Do forward pass
        cf,_,_ = self.forward_prop(model,x)
        # FDDN FlowRates
        V_true_LTR, V_hat_FDDN, _ = self.FDDN_SolverProcess(cf, None, row_id, dia, MTR_DuctArea, zeta_col_names, target_df)
        delta_V = V_hat_FDDN[0] - V_true_LTR[0]
        return cf, V_true_LTR, V_hat_FDDN, delta_V
