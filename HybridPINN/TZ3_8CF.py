## This script contains the Hybrid PINN class with adaptive Line search
## Version Description:
    ## In this version, the neural network outputs 8 correction factors.
    ## This version implements code to automatically configure the neural network weights and other procesess based on the passed hl_dim (hidden layer dimensions) list.
## Author: Dilip Rajkumar

import pandas as pd
import numpy as np
from itertools import chain
from copy import deepcopy
import sys
from scipy import optimize
from TZ3_FDDN_Lib import fddn_zeta_flag_input, FDDN_Solver, FDDN_output_df_gen, zeta_values

# A random seed is used to reproduce the weight initialization values and training performance everytime this scipt is run.
np.random.seed(13)

MTR = 'R300_HD'
zeta_select_elements = ['MTR-300', 'riser_LAO_RHS', 'riser_LAO_LHS', 'riser_CAO_RHS', 'riser_CAO_LHS','R-310', 'R-311', 'R-312', 'R-313','CAOR-320_fwd', 'CAOR-320_mid', 'CAOR-321_fwd', 'CAOR-321_mid','3302.2-3305','2301-3302.1','2301-3302.2']
fm_flag_sel_elements = ['LAO-330', 'LAO-331', 'LAO-332', 'LAO-333', 'LAO-334', 'LAO-335']

# Define custom MSE function
MSE = lambda y_hat,y_true:  np.mean((np.asarray(y_hat)-np.asarray(y_true))**2)

## Calculate Loss Coefficients
def SHR_Zeta_3D(hole_dia, MTR_DuctArea, cf):
    '''
    Computes the Zeta with 3D Correction Factor (cf) for Single Hole Retrictors
    '''
    n_holes = 1 # MTR is a single hole restrictor
    MTR_New_Area = n_holes * (np.pi/4) * (hole_dia / 1000)**2 # Divide dia by 1000 to convert mm to m
    f0_f1 = MTR_New_Area/MTR_DuctArea
    l_cross = 1/hole_dia
    zeta_dash = 0.13 + 0.34 * 10**(-(3.4 * l_cross + 88.4 * l_cross**2.3))
    zeta_SHR_1D = ((1 - f0_f1 + (zeta_dash**0.5) * (1 - f0_f1)**0.375)**2) * f0_f1**(-2) # 1D Zeta
    zeta_SHR_3D = zeta_SHR_1D * cf  # Zeta with 3D Correction Factor
    return MTR_New_Area, zeta_SHR_3D

def MHR_Zeta_3D(nr_holes, MHR_DuctArea, cf):
    '''
    Computes the Zeta with 3D Correction Factor (cf) for Multi Hole Restrictors
    '''
    hole_dia = 8 # All MHRs in TZ3 have hole_dia = 8
    MHR_New_Area = nr_holes * (np.pi/4) * (hole_dia / 1000)**2 # Divide dia by 1000 to convert mm to m
    f0_f1 = MHR_New_Area/MHR_DuctArea
    l_cross = (0.00144*1000)/hole_dia
    phi = 0.25 + (0.535 * l_cross**8) / (0.05 + l_cross**7)
    tau = (2.4 - l_cross) * 10**(-phi)
    zeta_MHR_1D = (0.5 * (1 - f0_f1)**0.75 + tau * (1 - f0_f1)**1.375 + (1 - f0_f1)**2 + 0.02 * l_cross) / f0_f1**2 # 1D Zeta
    zeta_MHR_3D = zeta_MHR_1D * cf  # Zeta with 3D Correction Factor
    return MHR_New_Area, zeta_MHR_3D

def riser_Zeta_3D(zeta_riser_CFD, cf):
    '''
    Computes the Zeta with 3D Correction Factor (cf) for riser ducts
    '''
    zeta_riser_1D = zeta_riser_CFD # 1D Zeta
    zeta_riser_3D = zeta_riser_1D * cf  # Zeta with 3D Correction Factor
    return zeta_riser_3D

def NormbyMax(df,features):
    '''Normalizes each column(feature) in a dataframe by the max value in that column'''
    df_scaled = pd.DataFrame()
    features_max_val = {}
    for e in features:
        max_val = df[e].max()
        features_max_val[e] = max_val
        df_scaled[e] = df[e] / max_val
        #print('scaling ',e,max_val,df[e],df_scaled[e])
    return features_max_val,df_scaled[features]

def pprint_dict(dictionary):
    '''Prints the contents of dictionary, its data type and shape for all the keys, line by line'''
    for k in dictionary.keys():
        print("{}:".format(k),list(dictionary[k]), type(dictionary[k]), "Shape:",dictionary[k].shape)

# Define Custom Class for Physics Informed Neural Network
class PINN(object):
    def __init__(self, espi_tuple, activation_function_choice = 'lrelu', error_loss_choice = 'full'):
        self.error_loss_choice = error_loss_choice
        self.MTR_epsi,self.riser_LAO_RHS_epsi, self.riser_LAO_LHS_epsi, self.riser_CAO_RHS_epsi, self.riser_CAO_LHS_epsi, self.riser_3302_2_3305_epsi, self.riser_3302_1_epsi, self.riser_3302_2_epsi = espi_tuple
        self.cf_lambda_reg_factor = 1000
        print('Activation Function selected:',activation_function_choice)
        print('Loss Function & Delta-k Used:',error_loss_choice)

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

    def initialize_parameters(self, input_nodes, hl_dim, output_nodes, X_train = None, seed_value = 42):
        print("SEED value:", seed_value)
        # print("X-Train:",X_train)
        print('Nr of Hidden layers:',len(hl_dim),';\tNr_Neurons in last hidden layer:', hl_dim[-1])
        np.random.seed(seed_value)

        # Initialize weights, biases and delta of those parameters for the input layer to first hidden layer
        w_i_dict = { 'W1': np.random.normal(0.0, input_nodes**-0.5,(input_nodes, hl_dim[0]))}
        b_i_dict = { 'b1': np.ones((hl_dim[0], ))}
        dw_i_dict = { 'dW1': np.zeros((input_nodes, hl_dim[0]))}
        db_i_dict = { 'db1': np.zeros((hl_dim[0],))}
        # Initialize empty dictionary for the hidden layers
        w_h_dict = {}
        b_h_dict = {}
        dw_h_dict = {}
        db_h_dict = {}
        # Initialize weights, biases and delta of those parameters for the last hidden layer to output layer
        w_o_dict = {'W{}'.format(len(hl_dim)+1): np.random.normal(0.0, output_nodes**-0.5,(hl_dim[-1], output_nodes))}
        b_o_dict = { 'b{}'.format(len(hl_dim)+1): np.ones((output_nodes, ))}
        dw_o_dict = {'dW{}'.format(len(hl_dim)+1): np.zeros((hl_dim[-1], output_nodes))}
        db_o_dict = {'db{}'.format(len(hl_dim)+1): np.zeros((output_nodes,))}

        # Construct dictionary for the hidden layers weight matrix
        for w_i,l_i in zip(range(2, len(hl_dim)+2), range(0, len(hl_dim)-1)):
            print('Initializing weights from first hidden layer to last hidden layer')
            w_h_dict["W{}".format(w_i)] = np.random.normal(0.0, hl_dim[l_i]**-0.5, (hl_dim[l_i], hl_dim[l_i + 1]))
            b_h_dict["b{}".format(w_i)] = np.ones((hl_dim[l_i + 1], ))
            dw_h_dict['dW{}'.format(w_i)] = np.zeros((hl_dim[l_i], hl_dim[l_i + 1]))
            db_h_dict['db{}'.format(w_i)] = np.zeros((hl_dim[l_i + 1],))

        # Package and return model parameters as a dictionary
        model =  dict(chain(w_i_dict.items(), b_i_dict.items(), w_h_dict.items(),b_h_dict.items(),w_o_dict.items(), b_o_dict.items()))
        print('FIRST - INITIAL WEIGHTS & BIASES:')
        pprint_dict(model)

        raw_cfs, _ , _ , _ = self.forward_prop(model, X_train)
        print('PRE-INITIALIZATION cf_array:',raw_cfs, type(raw_cfs))
        delta_b = 1 - np.asarray(raw_cfs)
        print('PRE-INITIALIZATION delta_b:',delta_b, type(delta_b), delta_b.shape)
        b_o_dict = { 'b{}'.format(len(hl_dim)+1): (np.ones((output_nodes, )) + delta_b) }

        model =  dict(chain(w_i_dict.items(), b_i_dict.items(), w_h_dict.items(),b_h_dict.items(),w_o_dict.items(), b_o_dict.items()))
        weight_gradients_initial = dict(chain(dw_i_dict.items(), dw_h_dict.items(), dw_o_dict.items()))
        bias_gradients_initial = dict(chain(db_i_dict.items(), db_h_dict.items(), db_o_dict.items()))
        print('\nFINAL - INITIAL WEIGHTS, BIASES and GRADIENTS:')
        pprint_dict(model), pprint_dict(weight_gradients_initial), pprint_dict(bias_gradients_initial)
        return model, weight_gradients_initial, bias_gradients_initial

    def cust_f_alpha(self, weight_gradients_old, bias_gradients_old, weight_gradients_new, bias_gradients_new):
        ## Get np array from dictionary
        weight_gradient_keys = [k for k, v in weight_gradients_old.items() if k.startswith('dW')]
        bias_gradient_keys = [k for k, v in bias_gradients_old.items() if k.startswith('db')]
        dw_arr_old = np.hstack([weight_gradients_old[k][0].flatten() for k in weight_gradient_keys]).flatten()
        db_arr_old = np.hstack([bias_gradients_old[k].flatten() for k in bias_gradient_keys]).flatten()
        dw_arr_new = np.hstack([weight_gradients_new[k][0].flatten() for k in weight_gradient_keys]).flatten()
        db_arr_new = np.hstack([bias_gradients_new[k].flatten() for k in bias_gradient_keys]).flatten()
        e0 = np.hstack((dw_arr_old,db_arr_old))
        e1 = np.hstack((dw_arr_new,db_arr_new))
        e0_mag = np.sqrt(np.dot(e0,e0))
        e1_mag = np.sqrt(np.dot(e1,e1))
        f_alpha = np.dot(e0, e1)
        # print('Magnitude of E0:',e0_mag)
        # print('Magnitude of E1:',e1_mag)
        # print('f.|E1|:',(f_alpha*e1_mag))
        return f_alpha

    def zfunc(self,lr,model0, weight_gradients0, bias_gradients0, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org):
        if type(idx) == int:
            _, _, weight_gradients1, bias_gradients1, _ , _ , _ = self.model_train(lr, model0, weight_gradients0, bias_gradients0, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
        else:
            print('zfunc - fullBatch length:',len(idx))
            _, _, weight_gradients1, bias_gradients1, _ , _    = self.model_train_batch(lr, model0, weight_gradients0, bias_gradients0 , X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)

        f_lr = self.cust_f_alpha(weight_gradients0, bias_gradients0, weight_gradients1, bias_gradients1)
        print('zfunc, lr, f_lr:', lr, f_lr)
        return f_lr

    def calc_lr2(self, f0, f1, lr0, lr1):
        lr2 = -(f0) / ((f1 - f0)/(lr1 - lr0))
        print('\nLearning Rate 2:', lr2)

        # Set min and max threshold limits for Learning rate 2
        if (lr2 < 0):
            print('Learning Rate 2 is negative')
        elif (lr2 > 1000):
            print('Learning Rate 2 is too high, so limiting it to 1000')
            lr2 = 1000 # if lr2 calculated exceeds +1000 set it to 1000
        return lr2

    def brents(self, f, x0, x1, fx0, fx1, model0, weight_gradients0, bias_gradients0, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org, max_iter, tolerance):
        # Modified brent line search algorithm
        # Expensive function value recalculations for fx0, fx1 and fx2 reduced to minimum
        # Modification to take into account that x1 < x0 is possible
        # Modification to take into account that root = 0 can happen

        break_flag = 0

        assert (fx0 * fx1) <= 0, "Root not bracketed"

        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0
            fx0, fx1 = fx1, fx0

        x2, fx2 = x0, fx0

        mflag = True
        steps_taken = 0

        root = x1

        while steps_taken < max_iter and abs(x1-x0) > tolerance:

            if fx0 != fx2 and fx1 != fx2:
                L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
                L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
                L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
                new = L0 + L1 + L2

            else:
                new = x1 - ( (fx1 * (x1 - x0)) / (fx1 - fx0) )

            sgn = np.sign(x1-x0)

            if ((new*sgn < ((3 * x0 + x1) / 4)*sgn or new*sgn > x1*sgn) or
                (mflag == True and (abs(new - x1)) >= (abs(x1 - x2) / 2)) or
                (mflag == False and (abs(new - x1)) >= (abs(x2 - d) / 2)) or
                (mflag == True and (abs(x1 - x2)) < tolerance) or
                (mflag == False and (abs(x2 - d)) < tolerance)):
                new = (x0 + x1) / 2
                mflag = True

            else:
                mflag = False

            fnew = f(new, model0, weight_gradients0, bias_gradients0,X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)

            d, x2 = x2, x1
            fd, fx2 = fx2, fx1

            if abs(fnew) < abs(fx1):
                root = new
            else:
                root = x1

            if (fx0 * fnew) < 0:
                x1 = new
                fx1 = fnew
            else:
                x0 = new
                fx0 = fnew

            if abs(fx0) < abs(fx1):
                x0, x1 = x1, x0
                fx0, fx1 = fx1, fx0

            if root == 0.0 and max_iter < 3:
                print('root == 0.0, max_iter', max_iter)
                max_iter = max_iter + 1

            steps_taken += 1

        if (root == 0.0):
            break_flag = 1
            print('|f0| < min(|f1|,|f2|) ! -> break', fx0, fx1, fnew)

        return root, steps_taken, break_flag

    def model_train(self, lr, model_old, weight_gradients_old, bias_gradients_old, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org):
        ''' Optimizes the recursive function calls used in the train functions'''
        model_new = self.update_parameters(lr, model_old, weight_gradients_old, bias_gradients_old)
        cfs, org_cf_tuple, epsi_cf_tuple, a_cache = self.forward_prop(model_new, X_train)
        V_true_LTR, V_hat_FDDN, V_hat_epsi_all_tuple = self.FDDN_SolverProcess(org_cf_tuple, epsi_cf_tuple, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df)
        MSE_FlowError = MSE(V_hat_FDDN.values, V_true_LTR.values)
        error, weight_gradients, bias_gradients = self.backward_prop(model_new, a_cache, X_train, np.reshape(cfs,(1,len(cfs))), V_max_org, V_true_LTR, V_hat_FDDN, V_hat_epsi_all_tuple)
        return MSE_FlowError, error, weight_gradients, bias_gradients, model_new, cfs, org_cf_tuple

    def train_1p_seq_LS(self, learning_rates, network_dim, feature_names, train_data, train_data_rparams, train_data_idx, restrictor_duct_areas, target_df, fddn_input_col_names, V_max_org, epochs=50):
        ''' Train the network on input features of one data point at a time sequentially till it attains convergence
            Arguments
            ---------
            learning_rates: Initial Learning Rates range to use for the Line Search
            network_dim:    nr. of neurons and layers of the network as a tuple
            feature_names:  list of input feature names as a list
            train_data:     2D array of input features, each row is one data record, each column is a feature
            train_data_rparams: tuple input of restrictor params (dia or nr_holes) for calculating Zeta for restrictors
            train_data_idx: indices of the training data in the current batch
            restrictor_duct_areas:    dictionary containing duct Areas for all the restrictors whose cfs are predicted
            target_df:      Source dataframe containing required columns for FDDN Process
            fddn_input_col_names: column names of all the elements in target_df to be passed as input to FDDN Process
            V_max_org:      Max Value of the input Flow Rate column(s)
            epochs:         nr. of epochs to train the model
        '''
        if len(fddn_input_col_names) == 0: # Check to see if Zeta_Col_names list is proper
            sys.exit('PROBLEM with ZETA COLUMN NAMES List. PLEASE CHECK INPUTS Properly in Jupyter Notebook')
        # Initialize list to store loss history
        N_i, hidden_layer_dim, output_nodes = network_dim
        #MTR_dia_train = train_data_rparams
        MTR_dia_train, riser_LAO_RHS_ZetaCFD,riser_LAO_LHS_ZetaCFD,riser_CAO_RHS_ZetaCFD,riser_CAO_LHS_ZetaCFD,riser_3302_2_3305_ZetaCFD,riser_3302_1_ZetaCFD,riser_3302_2_ZetaCFD = train_data_rparams
        MSE_flowloss_hist_train = []
        error_loss_hist_train = []
        # lr2_hist  = []
        losses_hist = {'HoV':[],'MSE_flowloss':[],'Error_loss':[]}
        cf_hist_train = {'HoV':[],'MTR_cf_hat':[],  'riser_LAO_RHS_cf_hat':[], 'riser_LAO_LHS_cf_hat':[], 'riser_CAO_RHS_cf_hat':[], 'riser_CAO_LHS_cf_hat':[], 'riser_3302_2_3305_cf_hat':[], 'riser_3302_1_cf_hat':[], 'riser_3302_2_cf_hat':[], 'Early_Stopping_Reason':[]}#, 'FlowRate_Diff_(LTR-FDDN)':[]}
        for idx in train_data_idx:
            X_train = train_data[feature_names].loc[[idx]].values
            HoV = target_df[['HoV']].loc[idx].values.tolist()
            lr0,lr1 = learning_rates[0], learning_rates[1]
            lr1_initial = learning_rates[1]
            print('\nBEGIN Neural Network training for {}'.format(HoV))
            restrictor_params = MTR_dia_train.loc[idx],riser_LAO_RHS_ZetaCFD.loc[idx],riser_LAO_LHS_ZetaCFD.loc[idx],riser_CAO_RHS_ZetaCFD.loc[idx],riser_CAO_LHS_ZetaCFD.loc[idx],riser_3302_2_3305_ZetaCFD.loc[idx],riser_3302_1_ZetaCFD.loc[idx],riser_3302_2_ZetaCFD.loc[idx]
            model_initial, weight_gradients_initial, bias_gradients_initial = self.initialize_parameters(N_i, hidden_layer_dim, output_nodes, X_train, seed_value = 13)
            for i in range(1,epochs):
                # print('INITIAL - MODEL PARAMETERS and GRADIENTS')
                # pprint_dict(model_initial), pprint_dict(weight_gradients_initial), pprint_dict(bias_gradients_initial)
                MSE_FlowError_lr0, error, weight_gradients0, bias_gradients0, model0, cfs, _ = self.model_train(lr0, model_initial, weight_gradients_initial, bias_gradients_initial, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                f0 = self.cust_f_alpha(weight_gradients0, bias_gradients0, weight_gradients0, bias_gradients0)
                print('f0:', f0, ';\tlr0:',lr0, ';\tMSE_FlowError:', MSE_FlowError_lr0, ';\tError:',error)
                ## Check error threshold for early stopping
                if (error < 0.1): #0.1
                    break_flag = 1
                    print('Error is very LOW (< 0.1):',(error))
                    break
                else:
                    break_flag = 0
                MSE_FlowError_lr1, error1, weight_gradients1, bias_gradients1, model1, cfs, org_cf_tuple1 = self.model_train(lr1, model0, weight_gradients0, bias_gradients0, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                f1 = self.cust_f_alpha(weight_gradients0, bias_gradients0, weight_gradients1, bias_gradients1)
                print('f1:', f1,';\tlr1:',lr1, ';\tMSE_FlowError:', MSE_FlowError_lr1, ';\tError:',error1)
                model_initial, weight_gradients_initial, bias_gradients_initial, MSE_FlowError, error, org_cf_tuple  = model1, weight_gradients1, bias_gradients1, MSE_FlowError_lr1, error1, org_cf_tuple1
                if (cfs < 0).any():
                    print('NEGATIVE Cfs FOUND at LR1 calculation -> Set lr1 = lr1 / 10 and go to next epoch')
                    lr1 = lr1 / 10
                    # Store Training history performance
                    MSE_flowloss_hist_train.append(MSE_FlowError)
                    error_loss_hist_train.append(error)
                    print('\nEPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train)
                    continue
                if (f1 < 0):
                    print('f1 < 0 -> Skip f2 and use Brent')
                    lr2, _, brent_break_flag = self.brents(self.zfunc, 0.0, lr1, f0, f1, model0, weight_gradients0, bias_gradients0, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org, max_iter = 1, tolerance =1e-5)
                    if (brent_break_flag == 1): # the intention of this is to end the for loop
                        break
                    #lr2 = optimize.brentq(self.zfunc, 0.0, lr1, args=(model0, weight_gradients0, bias_gradients0,X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org), maxiter=2, disp=False) # xtol=1.0e-08,
                    MSE_FlowError_lr2, error2, weight_gradients2, bias_gradients2, model2, cfs, org_cf_tuple2 = self.model_train(lr2, model0, weight_gradients0, bias_gradients0, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                    model_initial, weight_gradients_initial, bias_gradients_initial, MSE_FlowError, error, org_cf_tuple  = model2, weight_gradients2, bias_gradients2, MSE_FlowError_lr2, error2, org_cf_tuple2
                    print('lr2 (Brent):',lr2, ';\tMSE_FlowError:', MSE_FlowError, ';\tError:',error)
                    # Store Training history performance
                    MSE_flowloss_hist_train.append(MSE_FlowError)
                    error_loss_hist_train.append(error)
                    print('\nEPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train)
                    # print('Model - WEIGHTS, BIASES and GRADIENTS')
                    # pprint_dict(model_initial)
                    continue
                lr2 = self.calc_lr2( f0, f1, lr0, lr1)
                if (lr2 <= 0):
                    print('lr2 <= 0 -> Set lr1 = lr1 * 5 and go to next epoch')
                    lr1 = lr1 * 5
                    # Store Training history performance
                    MSE_flowloss_hist_train.append(MSE_FlowError_lr1)
                    error_loss_hist_train.append(error1)
                    print('\nEPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train)
                    continue
                MSE_FlowError_lr2, error2, weight_gradients2, bias_gradients2, model2, cfs, org_cf_tuple2 = self.model_train(lr2, model0, weight_gradients0, bias_gradients0, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                if (cfs < 0).any():
                    print('lr2:',lr2, ';\tMSE_FlowError:', MSE_FlowError_lr2, ';\tError:',error2)
                    print('NEGATIVE Cfs FOUND at LR2 calculation. USING WEIGHTS and BIASES from LR1 Calculation')
                    print('lr1:',lr1, ';\tMSE_FlowError:', MSE_FlowError_lr1, ';\tError:',error1)
                    model_initial, weight_gradients_initial, bias_gradients_initial, MSE_FlowError, error, org_cf_tuple  = model1, weight_gradients1, bias_gradients1, MSE_FlowError_lr1, error1, org_cf_tuple1
                    # Store Training history performance
                    MSE_flowloss_hist_train.append(MSE_FlowError)
                    error_loss_hist_train.append(error)
                    print('\nEPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train)
                    # print('Model - WEIGHTS, BIASES and GRADIENTS')
                    # pprint_dict(model_initial)
                    continue
                f2 = self.cust_f_alpha(weight_gradients0, bias_gradients0, weight_gradients2, bias_gradients2)
                print('f2:', f2,';\tlr2:',lr2, ';\tMSE_FlowError:', MSE_FlowError_lr2, ';\tError:',error2)

                if (abs(f2) > abs(f1)):
                    print('|f2| > |f1| -> Set lr1 = lr1 * 5 and go to next epoch')
                    lr1 = lr1 * 5
                    model_initial, weight_gradients_initial, bias_gradients_initial, MSE_FlowError, error, org_cf_tuple  = model1, weight_gradients1, bias_gradients1, MSE_FlowError_lr1, error1, org_cf_tuple1
                    # Store Training history performance
                    MSE_flowloss_hist_train.append(MSE_FlowError_lr1)
                    error_loss_hist_train.append(error1)
                    print('\nEPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train)
                    continue
                if (f2 < 0):
                    print('f2 < 0 -> Use Brent')
                    lr2_brent, _, brent_break_flag = self.brents(self.zfunc, lr2, lr1, f2, f1, model0, weight_gradients0, bias_gradients0, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org, max_iter = 1, tolerance=1e-5)
                    if (brent_break_flag == 1):
                        break
                    #lr2_brent = optimize.brentq(self.zfunc, lr2, lr1, args=(model0, weight_gradients0, bias_gradients0,X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org), maxiter=1, disp=False) # xtol=1.0e-08,
                    lr2 = lr2_brent
                    MSE_FlowError_lr2, error2, weight_gradients2, bias_gradients2, model2, cfs, org_cf_tuple2 = self.model_train(lr2, model0, weight_gradients0, bias_gradients0, X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                else:
                    print('f2 >= 0 -> Use lr2')
                model_initial, weight_gradients_initial, bias_gradients_initial, MSE_FlowError, error, org_cf_tuple  = model2, weight_gradients2, bias_gradients2, MSE_FlowError_lr2, error2, org_cf_tuple2
                print('lr2:',lr2, ';\tMSE_FlowError:', MSE_FlowError, ';\tError:',error)

                if(lr1 != lr1_initial):
                    lr1 = np.sqrt(lr1*lr1_initial)
                    print('lr1 changed for future epochs from ', lr1_initial, 'to ', lr1)
                    lr1_initial = lr1
                # print('Model2 - WEIGHTS, BIASES and GRADIENTS')
                # pprint_dict(model2),pprint_dict(weight_gradients2),pprint_dict(bias_gradients2)

                # Store Error and Train Loss History
                MSE_flowloss_hist_train.append(MSE_FlowError)
                error_loss_hist_train.append(error)
                # Print loss
                print('\nEPOCH_l:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train)


            early_stopping_bf = self.early_stopping(error_loss_hist_train, 10) # Call early stopping function to monitor last 10 epochs of the loss array

            #print('\nEPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",np.around(MSE_flowloss_hist_train[i-1],7), ";\tError_loss (Train): ", np.around(error_loss_hist_train[i-1],7))
            if (break_flag == 1) or (i == (epochs - 1)) or (early_stopping_bf == 1):
                MTR_cf, riser_LAO_RHS_cf, riser_LAO_LHS_cf, riser_CAO_RHS_cf, riser_CAO_LHS_cf, riser_3302_2_3305_cf, riser_3302_1_cf, riser_3302_2_cf = org_cf_tuple
                print('Terminating Neural Network Training for {}'.format(HoV[0]))
                # print('break_flag:',break_flag, ';\ti:', i, ';\tearly_stopping_bf:', early_stopping_bf)
                print('MSEFlow Loss History for {}:'.format(HoV[0]),MSE_flowloss_hist_train)
                print('Error Loss History for {}:'.format(HoV[0]),error_loss_hist_train)
                losses_hist['HoV'].append(HoV[0])
                losses_hist['MSE_flowloss'].append(MSE_flowloss_hist_train)
                losses_hist['Error_loss'].append(error_loss_hist_train)
                cf_hist_train['HoV'].append(HoV[0])
                cf_hist_train['MTR_cf_hat'].append(MTR_cf)
                cf_hist_train['riser_LAO_RHS_cf_hat'].append(riser_LAO_RHS_cf)
                cf_hist_train['riser_LAO_LHS_cf_hat'].append(riser_LAO_LHS_cf)
                cf_hist_train['riser_CAO_RHS_cf_hat'].append(riser_CAO_RHS_cf)
                cf_hist_train['riser_CAO_LHS_cf_hat'].append(riser_CAO_LHS_cf)
                cf_hist_train['riser_3302_2_3305_cf_hat'].append(riser_3302_2_3305_cf)
                cf_hist_train['riser_3302_1_cf_hat'].append(riser_3302_1_cf)
                cf_hist_train['riser_3302_2_cf_hat'].append(riser_3302_2_cf)
                MSE_flowloss_hist_train = []
                error_loss_hist_train = []
                # cf_hist_train['FlowRate_Diff_(LTR-FDDN)'].append(V_true_LTR[0] - V_hat_FDDN[0] )
                if (break_flag == 1):
                    print('EARLY STOPPING ACTIVATED - Error Converged')
                    cf_hist_train['Early_Stopping_Reason'].append('ErrorConverged')
                elif (early_stopping_bf == 1) or (brent_break_flag == 1):
                    print('EARLY STOPPING ACTIVATED - Error Loss Stagnated')
                    cf_hist_train['Early_Stopping_Reason'].append('ErrorLossStagned')
                elif (i == (epochs - 1)):
                    print('Set Nr. of Epochs Reached')
                    cf_hist_train['Early_Stopping_Reason'].append('SetEpochsReached')
                elif (idx == train_data_idx[-1]):
                    print('TRAINING PROCESS COMPLETED - All HoVs Converged')
                # break

        return MSE_flowloss_hist_train, error_loss_hist_train, cf_hist_train, losses_hist


    def model_train_batch(self, lr, model_old, weight_gradients_old , bias_gradients_old , X_train, batch_idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org):
        V_true_LTR_arr, V_hat_FDDN_arr, cf_array = [], [], []
        V_hat_MTR_epsi_FDDN_arr, V_hat_riser_LAO_RHS_epsi_FDDN_arr, V_hat_riser_LAO_LHS_epsi_FDDN_arr, V_hat_riser_CAO_RHS_epsi_FDDN_arr, V_hat_riser_CAO_LHS_epsi_FDDN_arr, V_hat_riser_3302_2_3305_epsi_FDDN_arr, V_hat_riser_3302_1_epsi_FDDN_arr, V_hat_riser_3302_2_epsi_FDDN_arr  = [], [], [], [], [], [], [], []
        model_new = self.update_parameters(lr, model_old, weight_gradients_old, bias_gradients_old)
        MTR_dia_arr, riser_LAO_RHS_ZetaCFD_arr, riser_LAO_LHS_ZetaCFD_arr, riser_CAO_RHS_ZetaCFD_arr, riser_CAO_LHS_ZetaCFD_arr, riser_3302_2_3305_ZetaCFD_arr, riser_3302_1_ZetaCFD_arr, riser_3302_2_ZetaCFD_arr  = restrictor_params
        weight_keys = [k for k, v in model_new.items() if k.startswith('W')]
        activations_array = {}
        activations_array.update({'a{}'.format(a_i):[] for a_i in range(0,len(weight_keys)+1)})
        for X, row_id, MTR_dia, riser_LAO_RHS_ZetaCFD, riser_LAO_LHS_ZetaCFD, riser_CAO_RHS_ZetaCFD, riser_CAO_LHS_ZetaCFD, riser_3302_2_3305_ZetaCFD, riser_3302_1_ZetaCFD, riser_3302_2_ZetaCFD in zip(X_train, batch_idx, MTR_dia_arr, riser_LAO_RHS_ZetaCFD_arr, riser_LAO_LHS_ZetaCFD_arr, riser_CAO_RHS_ZetaCFD_arr, riser_CAO_LHS_ZetaCFD_arr, riser_3302_2_3305_ZetaCFD_arr, riser_3302_1_ZetaCFD_arr, riser_3302_2_ZetaCFD_arr):
            r_param = (MTR_dia, riser_LAO_RHS_ZetaCFD, riser_LAO_LHS_ZetaCFD, riser_CAO_RHS_ZetaCFD, riser_CAO_LHS_ZetaCFD, riser_3302_2_3305_ZetaCFD, riser_3302_1_ZetaCFD, riser_3302_2_ZetaCFD)
            # Forward propagation
            cfs, org_cf_tuple , epsi_cf_tuple , a_cache = self.forward_prop(model_new, X)
            for k in activations_array.keys():
                activations_array[k].append(a_cache[k])
            V_true_LTR, V_hat_FDDN, V_hat_epsi_tuple = self.FDDN_SolverProcess(org_cf_tuple, epsi_cf_tuple, row_id, r_param, restrictor_duct_areas, fddn_input_col_names, target_df)
            V_hat_MTR_epsi_FDDN, V_hat_riser_LAO_RHS_epsi_FDDN, V_hat_riser_LAO_LHS_epsi_FDDN, V_hat_riser_CAO_RHS_epsi_FDDN, V_hat_riser_CAO_LHS_epsi_FDDN, V_hat_riser_3302_2_3305_epsi_FDDN, V_hat_riser_3302_1_epsi_FDDN, V_hat_riser_3302_2_epsi_FDDN = V_hat_epsi_tuple
            ## Calculate Percentage Difference between flowrates for early stopping
            percentage_diff = 100 * 2 * (abs(V_true_LTR.values - V_hat_FDDN.values)) / (V_true_LTR.values + V_hat_FDDN.values)
            if (np.linalg.norm(percentage_diff) < 2): # Magnitude of percentage difference vector < x
                print('TRAIN DATA - % Diff in Flowrates is LOW:', percentage_diff)
            # Append elements into a list
            V_true_LTR_arr.append(V_true_LTR)
            V_hat_FDDN_arr.append(V_hat_FDDN)
            V_hat_MTR_epsi_FDDN_arr.append(V_hat_MTR_epsi_FDDN)
            V_hat_riser_LAO_RHS_epsi_FDDN_arr.append(V_hat_riser_LAO_RHS_epsi_FDDN)
            V_hat_riser_LAO_LHS_epsi_FDDN_arr.append(V_hat_riser_LAO_LHS_epsi_FDDN)
            V_hat_riser_CAO_RHS_epsi_FDDN_arr.append(V_hat_riser_CAO_RHS_epsi_FDDN)
            V_hat_riser_CAO_LHS_epsi_FDDN_arr.append(V_hat_riser_CAO_LHS_epsi_FDDN)
            V_hat_riser_3302_2_3305_epsi_FDDN_arr.append(V_hat_riser_3302_2_3305_epsi_FDDN)
            V_hat_riser_3302_1_epsi_FDDN_arr.append(V_hat_riser_3302_1_epsi_FDDN)
            V_hat_riser_3302_2_epsi_FDDN_arr.append(V_hat_riser_3302_2_epsi_FDDN)
            cf_array.append(list(org_cf_tuple))

        V_true_LTR_df = pd.concat(V_true_LTR_arr)
        V_hat_FDDN_df = pd.concat(V_hat_FDDN_arr)
        MSE_FlowError = MSE(V_true_LTR_df.values, V_hat_FDDN_df.values)
        V_hat_MTR_epsi_FDDN_df    = pd.concat(V_hat_MTR_epsi_FDDN_arr)
        V_hat_riser_LAO_RHS_epsi_FDDN_df = pd.concat(V_hat_riser_LAO_RHS_epsi_FDDN_arr)
        V_hat_riser_LAO_LHS_epsi_FDDN_df = pd.concat(V_hat_riser_LAO_LHS_epsi_FDDN_arr)
        V_hat_riser_CAO_RHS_epsi_FDDN_df = pd.concat(V_hat_riser_CAO_RHS_epsi_FDDN_arr)
        V_hat_riser_CAO_LHS_epsi_FDDN_df = pd.concat(V_hat_riser_CAO_LHS_epsi_FDDN_arr)
        V_hat_riser_3302_2_3305_epsi_FDDN_df = pd.concat(V_hat_riser_3302_2_3305_epsi_FDDN_arr)
        V_hat_riser_3302_1_epsi_FDDN_df = pd.concat(V_hat_riser_3302_1_epsi_FDDN_arr)
        V_hat_riser_3302_2_epsi_FDDN_df = pd.concat(V_hat_riser_3302_2_epsi_FDDN_arr)
        V_hat_epsi_FDDN_all = (V_hat_MTR_epsi_FDDN_df, V_hat_riser_LAO_RHS_epsi_FDDN_df, V_hat_riser_LAO_LHS_epsi_FDDN_df, V_hat_riser_CAO_RHS_epsi_FDDN_df, V_hat_riser_CAO_LHS_epsi_FDDN_df, V_hat_riser_3302_2_3305_epsi_FDDN_df, V_hat_riser_3302_1_epsi_FDDN_df, V_hat_riser_3302_2_epsi_FDDN_df)
        error, weight_gradients_new, bias_gradients_new = self.backward_prop(model_new, activations_array, X_train, np.asarray(cf_array), V_max_org, V_true_LTR_df, V_hat_FDDN_df, V_hat_epsi_FDDN_all )
        # print('LTR DataFrame:\n',V_true_LTR_df)
        # print('FDDN DataFrame:\n',V_hat_FDDN_df)
        # print('CF Appended Arr:',cf_array, len(cf_array))
        # print('V_hat_MTR_epsi_FDDN_arr:\n',V_hat_MTR_epsi_FDDN_arr, len(V_hat_MTR_epsi_FDDN_arr))
        # print('V_hat_riser_CAO_LHS_epsi_FDDN_arr:\n',V_hat_riser_CAO_LHS_epsi_FDDN_arr, len(V_hat_riser_CAO_LHS_epsi_FDDN_arr))
        return MSE_FlowError, error, weight_gradients_new, bias_gradients_new, model_new, cf_array

    def val_data_batch(self, i, epochs, cf_hist_val, val_data_rparams_tuple, val_data_idx, val_batch_size, val_data, model, feature_names, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org):
        ## Evaluate Validation dataset
        val_MTR_dia, val_riser_LAO_RHS_ZetaCFD, val_riser_LAO_LHS_ZetaCFD, val_riser_CAO_RHS_ZetaCFD, val_riser_CAO_LHS_ZetaCFD, val_riser_3302_2_3305_ZetaCFD, val_riser_3302_1_ZetaCFD, val_riser_3302_2_ZetaCFD = val_data_rparams_tuple
        if (val_data_idx != None):
            V_LTR_val_arr, V_FDDN_val_arr, cf_val_arr, flow_delta_val = [], [], [], []
            if (len(val_data_idx) == val_batch_size):
                val_batch_idx = val_data_idx
                print('\nUSING ENTIRE VALIDATION DATASET in current Batch:', val_batch_idx)
            else:
                val_batch_idx = np.random.choice(val_data_idx, replace = False, size = val_batch_size)
                print('\nIndicies Selected in VALIDATION Batch:',val_batch_idx)
            X_val = val_data[feature_names].loc[val_batch_idx].values
            MTR_dia_val = val_MTR_dia.loc[val_batch_idx]
            riser_LAO_RHS_Zeta_val = val_riser_LAO_RHS_ZetaCFD.loc[val_batch_idx]
            riser_LAO_LHS_Zeta_val = val_riser_LAO_LHS_ZetaCFD.loc[val_batch_idx]
            riser_CAO_RHS_Zeta_val = val_riser_CAO_RHS_ZetaCFD.loc[val_batch_idx]
            riser_CAO_LHS_Zeta_val = val_riser_CAO_LHS_ZetaCFD.loc[val_batch_idx]
            riser_3302_2_3305_Zeta_val = val_riser_3302_2_3305_ZetaCFD.loc[val_batch_idx]
            riser_3302_1_Zeta_val = val_riser_3302_1_ZetaCFD.loc[val_batch_idx]
            riser_3302_2_Zeta_val = val_riser_3302_2_ZetaCFD.loc[val_batch_idx]
            for X, row_idx, MTR_dia, riser_LAO_RHS_Zeta, riser_LAO_LHS_Zeta, riser_CAO_RHS_Zeta, riser_CAO_LHS_Zeta, riser_3302_2_3305_Zeta, riser_3302_1_Zeta, riser_3302_2_Zeta in zip(X_val, val_batch_idx, MTR_dia_val, riser_LAO_RHS_Zeta_val, riser_LAO_LHS_Zeta_val, riser_CAO_RHS_Zeta_val, riser_CAO_LHS_Zeta_val, riser_3302_2_3305_Zeta_val, riser_3302_1_Zeta_val, riser_3302_2_Zeta_val):
                rparams_val = (MTR_dia, riser_LAO_RHS_Zeta, riser_LAO_LHS_Zeta, riser_CAO_RHS_Zeta, riser_CAO_LHS_Zeta, riser_3302_2_3305_Zeta, riser_3302_1_Zeta, riser_3302_2_Zeta)
                # Forward propagation
                cf_val, V_LTR, V_FDDN, delta_V = self.predict(model, X, row_idx, rparams_val, restrictor_duct_areas, fddn_input_col_names, target_df)
                MTR_cf_val, riser_LAO_RHS_cf_val, riser_LAO_LHS_cf_val, riser_CAO_RHS_cf_val, riser_CAO_LHS_cf_val, riser_3302_2_3305_cf_val, riser_3302_1_cf_val, riser_3302_2_cf_val = cf_val
                hov_val = target_df[['HoV']].loc[row_idx].values.tolist()
                cf_val_arr.append(list(cf_val))
                V_LTR_val_arr.append(V_LTR)
                V_FDDN_val_arr.append(V_FDDN)
                flow_delta_val.append(np.absolute(delta_V.values))
                percentage_diff_val = 100 * 2 * (abs(V_LTR.values - V_FDDN.values)) / (V_LTR.values + V_FDDN.values)
                if (np.linalg.norm(percentage_diff_val) < 2) or (i == (epochs - 1)):
                    print('VALIDATION DATA - % Diff in Flowrates is LOW:',percentage_diff_val,'%')
                    cf_hist_val['HoV'].append(hov_val[0])
                    cf_hist_val['MTR_cf_hat_val'].append(MTR_cf_val)
                    cf_hist_val['riser_LAO_RHS_cf_hat_val'].append(riser_LAO_RHS_cf_val)
                    cf_hist_val['riser_LAO_LHS_cf_hat_val'].append(riser_LAO_LHS_cf_val)
                    cf_hist_val['riser_CAO_RHS_cf_hat_val'].append(riser_CAO_RHS_cf_val)
                    cf_hist_val['riser_CAO_LHS_cf_hat_val'].append(riser_CAO_LHS_cf_val)
                    cf_hist_val['riser_3302_2_3305_cf_hat_val'].append(riser_3302_2_3305_cf_val)
                    cf_hist_val['riser_3302_1_cf_hat_val'].append(riser_3302_1_cf_val)
                    cf_hist_val['riser_3302_2_cf_hat_val'].append(riser_3302_2_cf_val)
                    cf_hist_val['Epoch'].append(i)
                    # cf_hist_val['FlowRate_Delta_(FDDN - LTR)'].append(delta_V)
            V_LTR_val_df = pd.concat(V_LTR_val_arr)
            V_FDDN_val_df = pd.concat(V_FDDN_val_arr)
            # Store validation data performance history
            val_error = self.cf_lambda_reg_factor * (1/(2 * val_batch_size)) * ( (1/V_max_org.values**2 * (V_LTR_val_df.values - V_FDDN_val_df.values)**2).sum() + (1/self.cf_lambda_reg_factor) * ((np.asarray(cf_val_arr) - 1)**2).sum() )  # Calculate val data error
            val_MSE_FlowError = MSE(V_LTR_val_df.values, V_FDDN_val_df.values)
            mean_flowdelta_val = np.mean(flow_delta_val)
        return val_error, val_MSE_FlowError, mean_flowdelta_val, cf_hist_val

    def train_batch_LS(self, learning_rates, network_dim, feature_names, train_data, train_data_rparams_tuple, train_data_idx, train_batch_size, restrictor_duct_areas, target_df, fddn_input_col_names, V_max_org, val_batch_size, val_data = None, val_data_rparams_tuple = None, val_data_idx = None,  epochs=50):
        ''' Train the network on input features of on a batch of points using minibatch gradient descent
            Arguments
            ---------
            learning_rates: Initial learning rates range for the Line Search
            network_dim:    nr. of neurons and layers of the network as a tuple
            feature_names:  list of input feature names as a list
            train_data:     2D array of input features, each row is one data record, each column is a feature
            train_data_rparams_tuple: tuple input of restrictor params (dia or nr_holes) for calculating Zeta for restrictors
            train_data_idx: indices of the training data in the current batch
            train_batch_size: no. of data points in the training batch
            restrictor_duct_areas:    dictionary containing duct Areas for all the restrictors whose cfs are predicted
            target_df:      Source dataframe containing required columns for FDDN Process
            fddn_input_col_names: Zeta names of all the elements
            V_max_org:      Max Value of the input Flow Rate column(s)
            val_data:       validation data points for making predictions at the end of every epochs
            val_data_rparams_tuple:   Tuple input of restrictor params (dia or nr_holes) for calculating Zeta for restrictors
            val_data_idx:   indices of validation data in the current batch
            val_batch_size: no. of data points in the validation batch
            epochs:         nr. of epochs to train the model
        '''
        if len(fddn_input_col_names) == 0:  # Check to see if Zeta_Col_names list is proper
            sys.exit('PROBLEM with ZETA COLUMN NAMES List. PLEASE CHECK INPUTS Properly in Jupyter Notebook')
        # Initialize list to store loss history
        N_i, hidden_layer_dim, output_nodes = network_dim
        lr0,lr1 = learning_rates[0], learning_rates[1]
        lr1_initial = lr1
        MTR_dia_train, riser_LAO_RHS_ZetaCFD_train, riser_LAO_LHS_ZetaCFD_train, riser_CAO_RHS_ZetaCFD_train, riser_CAO_LHS_ZetaCFD_train, riser_3302_2_3305_ZetaCFD_train, riser_3302_1_ZetaCFD_train, riser_3302_2_ZetaCFD_train = train_data_rparams_tuple
        MSE_flowloss_hist_train = []
        error_loss_hist_train = []
        MSE_flowloss_hist_val = []
        error_loss_hist_val = []
        flow_delta_val_mean = []
        cf_hist_train = {'HoV':[],'MTR_cf_hat':[],'riser_LAO_RHS_cf_hat':[], 'riser_LAO_LHS_cf_hat':[], 'riser_CAO_RHS_cf_hat':[], 'riser_CAO_LHS_cf_hat':[], 'riser_3302_2_3305_cf_hat':[], 'riser_3302_1_cf_hat':[], 'riser_3302_2_cf_hat':[], 'Epoch':[]}
        cf_hist_val = {'HoV':[],'MTR_cf_hat_val':[],'riser_LAO_RHS_cf_hat_val':[], 'riser_LAO_LHS_cf_hat_val':[], 'riser_CAO_RHS_cf_hat_val':[], 'riser_CAO_LHS_cf_hat_val':[], 'riser_3302_2_3305_cf_hat_val':[], 'riser_3302_1_cf_hat_val':[], 'riser_3302_2_cf_hat_val':[], 'Epoch':[]}
        if 0 in train_data_idx:
            zero_idx_loc = train_data_idx.index(0)
        if (len(train_data_idx) == train_batch_size):
            batch_idx = train_data_idx
            print('\nUSING ENTIRE TRAIN DATASET in current TRAINING Batch:', batch_idx)
        else:
            batch_idx = np.random.choice(train_data_idx, replace = False, size = train_batch_size)
            print('\nIndicies Selected in Current TRAINING Batch:',batch_idx)
        X_train = train_data[feature_names].loc[batch_idx].values
        MTR_dia = MTR_dia_train.loc[batch_idx]
        riser_LAO_RHS_ZetaCFD = riser_LAO_RHS_ZetaCFD_train.loc[batch_idx]
        riser_LAO_LHS_ZetaCFD = riser_LAO_LHS_ZetaCFD_train.loc[batch_idx]
        riser_CAO_RHS_ZetaCFD = riser_CAO_RHS_ZetaCFD_train.loc[batch_idx]
        riser_CAO_LHS_ZetaCFD = riser_CAO_LHS_ZetaCFD_train.loc[batch_idx]
        riser_3302_2_3305_ZetaCFD = riser_3302_2_3305_ZetaCFD_train.loc[batch_idx]
        riser_3302_1_ZetaCFD = riser_3302_1_ZetaCFD_train.loc[batch_idx]
        riser_3302_2_ZetaCFD = riser_3302_2_ZetaCFD_train.loc[batch_idx]
        restrictor_params = (MTR_dia,  riser_LAO_RHS_ZetaCFD, riser_LAO_LHS_ZetaCFD, riser_CAO_RHS_ZetaCFD, riser_CAO_LHS_ZetaCFD, riser_3302_2_3305_ZetaCFD, riser_3302_1_ZetaCFD, riser_3302_2_ZetaCFD)
        # Initialize model
        model_initial, weight_gradients_initial, bias_gradients_initial = self.initialize_parameters(N_i, hidden_layer_dim, output_nodes, train_data[feature_names].loc[batch_idx[0]].values, seed_value = 13)
        for i in range(0, epochs): # Gradient descent - Loop over epochs
            # print('INITIAL MODEL - WEIGHTS, BIASES and GRADIENTS')
            # pprint_dict(model_initial), pprint_dict(weight_gradients_initial), pprint_dict(bias_gradients_initial)
            MSE_FlowError_lr0, error0, weight_gradients0, bias_gradients0, model0, cf_array_train = self.model_train_batch(lr0, model_initial, weight_gradients_initial, bias_gradients_initial, X_train, batch_idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
            # print('\nMODEL0 - Weights and Biases') pprint_dict(model0)
            f0 = self.cust_f_alpha(weight_gradients0, bias_gradients0, weight_gradients0, bias_gradients0)
            print('f0:', f0, ';\tlr0:',lr0, ';\tMSE_FlowError:', MSE_FlowError_lr0, ';\tError:',error0)
            MSE_FlowError_lr1, error1, weight_gradients1, bias_gradients1, model1, cf_array_train1 = self.model_train_batch(lr1, model0, weight_gradients0, bias_gradients0, X_train, batch_idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
            f1 = self.cust_f_alpha(weight_gradients0, bias_gradients0, weight_gradients1, bias_gradients1)
            print('f1:', f1,';\tlr1:',lr1, ';\tMSE_FlowError:', MSE_FlowError_lr1, ';\tError:',error1)
            if (np.array(cf_array_train1) < 0).any():
                print('NEGATIVE Cfs FOUND at LR1 calculation. -> Set lr1 = lr1 / 10 and go to next epoch')
                lr1 = lr1 / 10
                # Store Training history performance
                MSE_flowloss_hist_train.append(MSE_FlowError_lr1)
                error_loss_hist_train.append(error1)
                val_error1, val_MSE_FlowError1, mean_flowdelta_val1, cf_hist_val = self.val_data_batch(i, epochs, cf_hist_val, val_data_rparams_tuple, val_data_idx, val_batch_size, val_data, model1, feature_names, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                MSE_flowloss_hist_val.append(val_MSE_FlowError1)
                error_loss_hist_val.append(val_error1)
                flow_delta_val_mean.append(mean_flowdelta_val1)
                print('EPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train, ";\tMSE Flowloss (Val): ", MSE_flowloss_hist_val, ";\tError_loss (Val): ",error_loss_hist_val)
                continue
            if (f1 < 0):
                print('f1 < 0 -> Skip f2 and use Brent')
                lr2, _, brent_break_flag =  self.brents(self.zfunc, 0.0, lr1, f0, f1, model0, weight_gradients0, bias_gradients0, X_train, batch_idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org, max_iter = 1, tolerance =1e-5)
                if (brent_break_flag == 1): # the intention of this is to end the for loop
                    break
                MSE_FlowError_lr2, error2, weight_gradients2, bias_gradients2, model2, cf_array_train2 = self.model_train_batch(lr2, model0, weight_gradients0, bias_gradients0, X_train, batch_idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                model_initial, weight_gradients_initial, bias_gradients_initial, MSE_FlowError, error, cf_array_train = model2, weight_gradients2, bias_gradients2, MSE_FlowError_lr2, error2, cf_array_train2
                print('lr2 (Brent):',lr2, ';\tMSE_FlowError:', MSE_FlowError, ';\tError:',error)
                # Store Training history performance
                MSE_flowloss_hist_train.append(MSE_FlowError)
                error_loss_hist_train.append(error)
                val_error, val_MSE_FlowError, mean_flowdelta_val, cf_hist_val = self.val_data_batch(i, epochs, cf_hist_val, val_data_rparams_tuple, val_data_idx, val_batch_size, val_data, model_initial, feature_names, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                MSE_flowloss_hist_val.append(val_MSE_FlowError)
                error_loss_hist_val.append(val_error)
                flow_delta_val_mean.append(mean_flowdelta_val)
                print('EPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train, ";\tMSE Flowloss (Val): ", MSE_flowloss_hist_val, ";\tError_loss (Val): ",error_loss_hist_val)
                continue
            lr2 = self.calc_lr2( f0, f1, lr0, lr1)
            if (lr2 <= 0):
                print('lr2 <= 0 -> Set lr1 = lr1 * 5 and go to next epoch')
                lr1 = lr1 * 5
                # Store Training history performance
                MSE_flowloss_hist_train.append(MSE_FlowError_lr1)
                error_loss_hist_train.append(error1)
                MSE_flowloss_hist_val.append(val_MSE_FlowError1)
                error_loss_hist_val.append(val_error1)
                flow_delta_val_mean.append(mean_flowdelta_val1)
                print('EPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train, ";\tMSE Flowloss (Val): ", MSE_flowloss_hist_val, ";\tError_loss (Val): ",error_loss_hist_val)
                continue
            MSE_FlowError_lr2, error2, weight_gradients2, bias_gradients2, model2, cf_array_train2 = self.model_train_batch(lr2, model0, weight_gradients0, bias_gradients0, X_train, batch_idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
            if (np.array(cf_array_train2) < 0).any():
                print('lr2:',lr2, ';\tMSE_FlowError:', MSE_FlowError_lr2, ';\tError:',error2)
                print('NEGATIVE Cfs FOUND at LR2 calculation. USING WEIGHTS and BIASES from LR1 Calculation')
                print('lr1:',lr1, ';\tMSE_FlowError:', MSE_FlowError_lr1, ';\tError:',error1)
                model_initial, weight_gradients_initial, bias_gradients_initial, MSE_FlowError, error, cf_array_train = model1, weight_gradients1, bias_gradients1, MSE_FlowError_lr1, error1, cf_array_train1
                # Store Training history performance
                MSE_flowloss_hist_train.append(MSE_FlowError)
                error_loss_hist_train.append(error)
                val_error, val_MSE_FlowError, mean_flowdelta_val, cf_hist_val = self.val_data_batch(i, epochs, cf_hist_val, val_data_rparams_tuple, val_data_idx, val_batch_size, val_data, model_initial, feature_names, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                MSE_flowloss_hist_val.append(val_MSE_FlowError)
                error_loss_hist_val.append(val_error)
                flow_delta_val_mean.append(mean_flowdelta_val)
                print('EPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train, ";\tMSE Flowloss (Val): ", MSE_flowloss_hist_val, ";\tError_loss (Val): ",error_loss_hist_val)
                continue
            f2 = self.cust_f_alpha(weight_gradients0, bias_gradients0, weight_gradients2, bias_gradients2)
            print('f2:', f2,';\tlr2:',lr2, ';\tMSE_FlowError:', MSE_FlowError_lr2, ';\tError:',error2)
            if (abs(f2) > abs(f1)):
                print('|f2| > |f1| -> Set lr1 = lr1 * 5 and go to next epoch')
                lr1 = lr1 * 5
                model_initial, weight_gradients_initial, bias_gradients_initial, MSE_FlowError, error, cf_array_train = model1, weight_gradients1, bias_gradients1, MSE_FlowError_lr1, error1, cf_array_train1
                # Store Training history performance
                MSE_flowloss_hist_train.append(MSE_FlowError)
                error_loss_hist_train.append(error)
                val_error, val_MSE_FlowError, mean_flowdelta_val, cf_hist_val = self.val_data_batch(i, epochs, cf_hist_val, val_data_rparams_tuple, val_data_idx, val_batch_size, val_data, model_initial, feature_names, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
                MSE_flowloss_hist_val.append(val_MSE_FlowError)
                error_loss_hist_val.append(val_error)
                flow_delta_val_mean.append(mean_flowdelta_val)
                print('EPOCH:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train, ";\tMSE Flowloss (Val): ", MSE_flowloss_hist_val, ";\tError_loss (Val): ",error_loss_hist_val)
                continue
            if (f2 < 0):
                print('f2 < 0 -> Use Brent')
                lr2_brent, _, brent_break_flag = self.brents(self.zfunc, lr2, lr1, f2, f1, model0, weight_gradients0, bias_gradients0, X_train, batch_idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org, max_iter = 1, tolerance=1e-5)
                if (brent_break_flag == 1):
                    break
                #lr2_brent = optimize.brentq(self.zfunc, lr2, lr1, args=(model0, weight_gradients0, bias_gradients0,X_train, idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org), maxiter=1, disp=False) # xtol=1.0e-08,
                lr2 = lr2_brent
                MSE_FlowError_lr2, error2, weight_gradients2, bias_gradients2, model2, cf_array_train2 = self.model_train_batch(lr2, model0, weight_gradients0, bias_gradients0, X_train, batch_idx, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
            else:
                print('f2 >= 0 -> Use lr2')
            model_initial, weight_gradients_initial, bias_gradients_initial, MSE_FlowError, error, cf_array_train  = model2, weight_gradients2, bias_gradients2, MSE_FlowError_lr2, error2, cf_array_train2
            print('lr2:',lr2, ';\tMSE_FlowError:', MSE_FlowError, ';\tError:',error)

            if(lr1 != lr1_initial):
                lr1 = np.sqrt(lr1*lr1_initial)
                print('lr1 changed for future epochs from ', lr1_initial, 'to ', lr1)
                lr1_initial = lr1
            # print('Model2 - WEIGHTS, BIASES and GRADIENTS')
            # pprint_dict(model2),pprint_dict(weight_gradients2),pprint_dict(bias_gradients2)

            # print('FINAL MODEL - WEIGHTS, BIASES and GRADIENTS')
            # pprint_dict(model_initial), pprint_dict(weight_gradients_initial),pprint_dict(bias_gradients_initial)
            MTR_cf_array = [item[0] for item in cf_array_train]
            riser_LAO_RHS_cf_array = [item[1] for item in cf_array_train]
            riser_LAO_LHS_cf_array = [item[2] for item in cf_array_train]
            riser_CAO_RHS_cf_array = [item[3] for item in cf_array_train]
            riser_CAO_LHS_cf_array = [item[4] for item in cf_array_train]
            riser_3302_2_3305_cf_array = [item[5] for item in cf_array_train]
            riser_3302_1_cf_array = [item[6] for item in cf_array_train]
            riser_3302_2_cf_array = [item[7] for item in cf_array_train]
            # Store Training history performance
            MSE_flowloss_hist_train.append(MSE_FlowError)
            error_loss_hist_train.append(error)
            val_error, val_MSE_FlowError, mean_flowdelta_val, cf_hist_val = self.val_data_batch(i, epochs, cf_hist_val, val_data_rparams_tuple, val_data_idx, val_batch_size, val_data, model_initial, feature_names, restrictor_duct_areas, fddn_input_col_names, target_df, V_max_org)
            MSE_flowloss_hist_val.append(val_MSE_FlowError)
            error_loss_hist_val.append(val_error)
            flow_delta_val_mean.append(mean_flowdelta_val)
            # Print loss
            print('Epoch:',i, ";\tProgress: {:2.1f}%".format(100 * i/float(epochs)),";\tMSE Flowloss (Train): ",MSE_flowloss_hist_train, ";\tError_loss (Train): ", error_loss_hist_train, ";\tMSE Flowloss (Val): ", MSE_flowloss_hist_val, ";\tError_loss (Val): ",error_loss_hist_val)

        # train_early_stopping_bf = self.early_stopping(error_loss_hist_train, 10) # Call early stopping function to monitor last 10 epochs of the train loss array
        # val_early_stopping_bf = self.early_stopping(error_loss_hist_val, 10) # Call early stopping function to monitor last 10 epochs of the validation loss array
        if (i == epochs - 1) : # if (train_early_stopping_bf == 1) or i == epochs - 1) :
            print('TRAINING PROCESS COMPLETED - Set Nr. of Training Epochs Reached')
            HoVs_train = target_df['HoV'].loc[batch_idx].values.tolist()
            cf_hist_train['Epoch'] = i
            cf_hist_train['HoV']                     = HoVs_train
            cf_hist_train['MTR_cf_hat']              = MTR_cf_array
            cf_hist_train['riser_LAO_RHS_cf_hat']    = riser_LAO_RHS_cf_array
            cf_hist_train['riser_LAO_LHS_cf_hat']    = riser_LAO_LHS_cf_array
            cf_hist_train['riser_CAO_RHS_cf_hat']    = riser_CAO_RHS_cf_array
            cf_hist_train['riser_CAO_LHS_cf_hat']    = riser_CAO_LHS_cf_array
            cf_hist_train['riser_3302_2_3305_cf_hat'] = riser_3302_2_3305_cf_array
            cf_hist_train['riser_3302_1_cf_hat']     = riser_3302_1_cf_array
            cf_hist_train['riser_3302_2_cf_hat']     = riser_3302_2_cf_array

        return model_initial, MSE_flowloss_hist_train, error_loss_hist_train, cf_hist_train, MSE_flowloss_hist_val, error_loss_hist_val, cf_hist_val, flow_delta_val_mean



    def forward_prop(self, model, X):
        '''
        model - Model parameters (i.e weights and biases)
        X    - activations from the input layer (basically the input features denoted by 'X')
        '''
        # Load parameters from model
        a_dict = {}
        z_dict = {}
        weights = [k for k, v in model.items() if k.startswith('W')]
        # print('X:', list(X), X.shape)
        # Create Empty dictionary with keys for the activations and linear combinations
        for i in range(0,len(weights)):
            a_dict.update({'a{}'.format(i):[]})
            z_dict.update({'z{}'.format(i+1):[]})

        # Assign first value in activations dict to features
        a_dict['a0'] = X
        for a_i, w_i in zip(range(0, len(weights)), range(1,len(weights)+1)):
            z_dict['z{}'.format(w_i)] = np.dot(a_dict['a{}'.format(a_i)], model['W{}'.format(w_i)]) + model['b{}'.format(w_i)] # adding biases
            a_dict['a{}'.format(a_i + 1)] = self.activation_function(z_dict['z{}'.format(w_i)])

        # Assign last value of activations dict to last z, because we want to use linear activations in the output layer
        a_dict['a{}'.format(len(weights))] = z_dict[list(z_dict.keys())[-1]]

        cfs = a_dict[list(a_dict.keys())[-1]] #[0] is added only for SingleHoVLoop mode
        cfs = cfs.ravel() #Flatten to 1D array
        # print('CFs:',cfs, cfs.shape)
        # Correction_factor output from final output layer having linear (identity) activation function
        MTR_cf_pred = cfs[0]
        riser_LAO_RHS_cf_pred = cfs[1]
        riser_LAO_LHS_cf_pred = cfs[2]
        riser_CAO_RHS_cf_pred = cfs[3]
        riser_CAO_LHS_cf_pred = cfs[4]
        riser_3302_2_3305_cf_pred = cfs[5]
        riser_3302_1_cf_pred = cfs[6]
        riser_3302_2_cf_pred = cfs[7]

        print('\nMTR-Cf:',MTR_cf_pred, '; riser_LAO_RHS-Cf:',riser_LAO_RHS_cf_pred,'; riser_LAO_LHS-Cf:',riser_LAO_LHS_cf_pred,'; riser_CAO_RHS-Cf:',riser_CAO_RHS_cf_pred,'; riser_CAO_LHS-Cf:',riser_CAO_LHS_cf_pred,'; riser_3302_2_3305-Cf:',riser_3302_2_3305_cf_pred,'; riser_3302_1-Cf:',riser_3302_1_cf_pred,'; riser_3302_2-Cf:',riser_3302_2_cf_pred)
        MTR_cf_epsi = MTR_cf_pred + self.MTR_epsi
        riser_LAO_RHS_cf_epsi = riser_LAO_RHS_cf_pred + self.riser_LAO_RHS_epsi
        riser_LAO_LHS_cf_epsi = riser_LAO_LHS_cf_pred + self.riser_LAO_LHS_epsi
        riser_CAO_RHS_cf_epsi = riser_CAO_RHS_cf_pred + self.riser_CAO_RHS_epsi
        riser_CAO_LHS_cf_epsi = riser_CAO_LHS_cf_pred + self.riser_CAO_LHS_epsi
        riser_3302_2_3305_cf_epsi = riser_3302_2_3305_cf_pred + self.riser_3302_2_3305_epsi
        riser_3302_1_cf_epsi = riser_3302_1_cf_pred + self.riser_3302_1_epsi
        riser_3302_2_cf_epsi = riser_3302_2_cf_pred + self.riser_3302_2_epsi

        org_cf_tuple = (MTR_cf_pred, riser_LAO_RHS_cf_pred, riser_LAO_LHS_cf_pred, riser_CAO_RHS_cf_pred, riser_CAO_LHS_cf_pred, riser_3302_2_3305_cf_pred,riser_3302_1_cf_pred,riser_3302_2_cf_pred)
        epsi_cf_tuple = (MTR_cf_epsi, riser_LAO_RHS_cf_epsi, riser_LAO_LHS_cf_epsi, riser_CAO_RHS_cf_epsi, riser_CAO_LHS_cf_epsi, riser_3302_2_3305_cf_epsi,riser_3302_1_cf_epsi,riser_3302_2_cf_epsi)

        return cfs, org_cf_tuple , epsi_cf_tuple , a_dict

    def early_stopping(self, train_loss, last_n_epochs = 10):
        '''Monitors the last n epochs of the train loss and returns a break flag to stop the training process
            Arguments
            ---------
            train_loss: array of the train loss values stored at the end of every training epoch
            last_n_epochs: the last 'n' epochs of the train_loss array to monitor for change in magnitude. Default value = 10
        '''
        if len(train_loss) >= last_n_epochs:
            # print("Error Loss List:", train_loss, ",...List Length:",len(train_loss))
            list_elements_diff = np.diff(train_loss[-last_n_epochs:])
            # print("Error list_elements_diff List:", list_elements_diff, ",...List Length:",len(list_elements_diff))
            loss_mag = np.linalg.norm(list_elements_diff)
            # print("Error loss_mag:", loss_mag, ",...List Length:",len(loss_mag))
        else:
            loss_mag = 1

        if (loss_mag < 0.01):
            print('NO CHANGE in TRAIN LOSS for the Last {} Epochs'.format(last_n_epochs))
            es_bf = 1
        else:
            es_bf = 0

        return es_bf

    def FDDN_SolverProcess(self, org_cf_tuple, epsi_cf_tuple, row_id, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df):
        '''
        Computes flowrates from updated zeta (loss coefficients) for restrictors for updated correction factors.
        '''
        print('Row ID:',row_id,'\tHoV:', target_df['HoV'].loc[[row_id]].values)
        MTR_cf, riser_LAO_RHS_cf, riser_LAO_LHS_cf, riser_CAO_RHS_cf, riser_CAO_LHS_cf, riser_3302_2_3305_cf, riser_3302_1_cf, riser_3302_2_cf = org_cf_tuple
        MTR_cf_epsi, riser_LAO_RHS_cf_epsi, riser_LAO_LHS_cf_epsi, riser_CAO_RHS_cf_epsi, riser_CAO_LHS_cf_epsi, riser_3302_2_3305_cf_epsi, riser_3302_1_cf_epsi, riser_3302_2_cf_epsi = epsi_cf_tuple
        MTR_dia,riser_LAO_RHS_ZetaCFD,riser_LAO_LHS_ZetaCFD,riser_CAO_RHS_ZetaCFD,riser_CAO_LHS_ZetaCFD,riser_3302_2_3305_ZetaCFD,riser_3302_1_ZetaCFD,riser_3302_2_ZetaCFD = restrictor_params
        ## Pass Neural network output (correction factor) to compute new Zeta and then re-run FDDN solver
        _, MTR_zeta_tuple = zip(*[SHR_Zeta_3D(MTR_dia,restrictor_duct_areas['R300_HD'],MTR_cf)])
        target_df[MTR+'_Zeta3D'].loc[row_id] = MTR_zeta_tuple[0] # Assign first element in zeta_tuple to required index
        riser_LAO_RHS_zeta = riser_Zeta_3D(riser_LAO_RHS_ZetaCFD, riser_LAO_RHS_cf)
        target_df['riser_LAO_RHS_Zeta3D'].loc[row_id] = riser_LAO_RHS_zeta
        riser_LAO_LHS_zeta = riser_Zeta_3D(riser_LAO_LHS_ZetaCFD, riser_LAO_LHS_cf)
        target_df['riser_LAO_LHS_Zeta3D'].loc[row_id] = riser_LAO_LHS_zeta
        riser_CAO_RHS_zeta = riser_Zeta_3D(riser_CAO_RHS_ZetaCFD, riser_CAO_RHS_cf)
        target_df['riser_CAO_RHS_Zeta3D'].loc[row_id] = riser_CAO_RHS_zeta
        riser_CAO_LHS_zeta = riser_Zeta_3D(riser_CAO_LHS_ZetaCFD, riser_CAO_LHS_cf)
        target_df['riser_CAO_LHS_Zeta3D'].loc[row_id] = riser_CAO_LHS_zeta
        riser_3302_2_3305_zeta = riser_Zeta_3D(riser_3302_2_3305_ZetaCFD, riser_3302_2_3305_cf)
        target_df['riser_3302_2_3305_Zeta3D'].loc[row_id] = riser_3302_2_3305_zeta
        riser_3302_1_zeta = riser_Zeta_3D(riser_3302_1_ZetaCFD, riser_3302_1_cf)
        target_df['riser_3302_1_Zeta3D'].loc[row_id] = riser_3302_1_zeta
        riser_3302_2_zeta = riser_Zeta_3D(riser_3302_2_ZetaCFD, riser_3302_2_cf)
        target_df['riser_3302_2_Zeta3D'].loc[row_id] = riser_3302_2_zeta
        #print('C H E C K  riser_3302_1_Zeta3D: ',riser_3302_1_ZetaCFD, riser_3302_1_cf)
        #print('C H E C K  riser_3302_2_Zeta3D: ',riser_3302_2_ZetaCFD, riser_3302_2_cf)
        #
        #print('zetas: MTR, riser_LAO_RHS_zeta, riser_LAO_LHS_zeta, riser_CAO_RHS_zeta, riser_CAO_LHS_zeta','riser_3302_1_zeta','riser_3302_2_zeta',MTR_zeta_tuple[0], riser_LAO_RHS_zeta, riser_LAO_LHS_zeta, riser_CAO_RHS_zeta, riser_CAO_LHS_zeta, riser_3302_1_zeta, riser_3302_2_zeta)
        mixp_input, ambp_input, ambt_input, zeta_input, fm_flag_input, fp2_flag_input = fddn_zeta_flag_input(target_df[['HoV','MIXP','AMBP','AMBT'] + fddn_input_col_names].loc[[row_id]])
        FDDN_FlowRates_raw_df = FDDN_Solver(zeta_select_elements, fm_flag_sel_elements, mixp_input, ambp_input, ambt_input, zeta_input, fm_flag_input, fp2_flag_input)
        FDDN_FlowRates_df = FDDN_output_df_gen(FDDN_FlowRates_raw_df )
        V_hat_FDDN = FDDN_FlowRates_df[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']]
        #print('V_hat_FDDN:',V_hat_FDDN.values)

        if (MTR_cf_epsi == None) & (riser_LAO_LHS_cf_epsi == None) & (riser_LAO_RHS_cf_epsi == None) & (riser_CAO_LHS_cf_epsi == None) & (riser_CAO_RHS_cf_epsi == None) & (riser_3302_2_3305_cf_epsi == None) & (riser_3302_1_cf_epsi == None) & (riser_3302_2_cf_epsi == None): # For validation and test points we won't compute epsi as there is no Backpropagation
            V_hat_epsi_all_tuple = tuple(np.zeros_like(epsi_cf_tuple))
        else: # if not all(epsi_cf_tuple): # For validation and test points we won't compute epsi as there is no Backpropagation
            ## Compute new flow rates for MTR-Epsi
            _, MTR_zeta_tuple_epsi = zip(*[SHR_Zeta_3D(MTR_dia, restrictor_duct_areas['R300_HD'], MTR_cf_epsi)])
            target_df[MTR+'_Zeta3D'].loc[row_id] = MTR_zeta_tuple_epsi[0] # Assign first element in zeta_tuple to required index
            target_df['riser_LAO_RHS_Zeta3D'].loc[row_id] = riser_LAO_RHS_zeta
            target_df['riser_LAO_LHS_Zeta3D'].loc[row_id] = riser_LAO_LHS_zeta
            target_df['riser_CAO_RHS_Zeta3D'].loc[row_id] = riser_CAO_RHS_zeta
            target_df['riser_CAO_LHS_Zeta3D'].loc[row_id] = riser_CAO_LHS_zeta
            target_df['riser_3302_2_3305_Zeta3D'].loc[row_id] = riser_3302_2_3305_zeta
            target_df['riser_3302_1_Zeta3D'].loc[row_id] = riser_3302_1_zeta
            target_df['riser_3302_2_Zeta3D'].loc[row_id] = riser_3302_2_zeta
            #
            #print('New_Zeta_epsi for MTR-300:' , MTR_zeta_tuple_epsi[0])
            #print('New_Zeta_epsi for MTR-300:' , target_df[MTR+'_Zeta3D'].loc[row_id])
            #
            mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi = fddn_zeta_flag_input(target_df[['HoV','MIXP','AMBP','AMBT']+fddn_input_col_names].loc[[row_id]])
            FDDN_FlowRates_raw_df_epsi = FDDN_Solver(zeta_select_elements, fm_flag_sel_elements, mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi)
            FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
            V_hat_MTR_epsi_FDDN = FDDN_FlowRates_df_epsi[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']]
            #print('V_hat_MTR_epsi_FDDN:',V_hat_MTR_epsi_FDDN.values)

            ## Compute new flow rates for riser_LAO_RHS_cf_epsi
            target_df[MTR+'_Zeta3D'].loc[row_id] = MTR_zeta_tuple[0]
            target_df['riser_LAO_RHS_Zeta3D'].loc[row_id] = riser_Zeta_3D(riser_LAO_RHS_ZetaCFD, riser_LAO_RHS_cf_epsi)
            target_df['riser_LAO_LHS_Zeta3D'].loc[row_id] = riser_LAO_LHS_zeta
            target_df['riser_CAO_RHS_Zeta3D'].loc[row_id] = riser_CAO_RHS_zeta
            target_df['riser_CAO_LHS_Zeta3D'].loc[row_id] = riser_CAO_LHS_zeta
            target_df['riser_3302_2_3305_Zeta3D'].loc[row_id] = riser_3302_2_3305_zeta
            target_df['riser_3302_1_Zeta3D'].loc[row_id] = riser_3302_1_zeta
            target_df['riser_3302_2_Zeta3D'].loc[row_id] = riser_3302_2_zeta
            #print('New_Zeta_epsi for riser_LAO_RHS:', target_df['riser_LAO_RHS_Zeta3D'].loc[row_id])
            mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi = fddn_zeta_flag_input(target_df[['HoV','MIXP','AMBP','AMBT']+fddn_input_col_names].loc[[row_id]])
            FDDN_FlowRates_raw_df_epsi = FDDN_Solver(zeta_select_elements, fm_flag_sel_elements, mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi)
            FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
            V_hat_riser_LAO_RHS_epsi_FDDN = FDDN_FlowRates_df_epsi[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']]
            #print('V_hat_MTR_epsi_FDDN:',V_hat_MTR_epsi_FDDN.values)

            ## Compute new flow rates for riser_LAO_LHS_cf_epsi
            target_df[MTR+'_Zeta3D'].loc[row_id] = MTR_zeta_tuple[0]
            target_df['riser_LAO_RHS_Zeta3D'].loc[row_id] = riser_LAO_RHS_zeta
            target_df['riser_LAO_LHS_Zeta3D'].loc[row_id] = riser_Zeta_3D(riser_LAO_LHS_ZetaCFD, riser_LAO_LHS_cf_epsi)
            target_df['riser_CAO_RHS_Zeta3D'].loc[row_id] = riser_CAO_RHS_zeta
            target_df['riser_CAO_LHS_Zeta3D'].loc[row_id] = riser_CAO_LHS_zeta
            target_df['riser_3302_2_3305_Zeta3D'].loc[row_id] = riser_3302_2_3305_zeta
            target_df['riser_3302_1_Zeta3D'].loc[row_id] = riser_3302_1_zeta
            target_df['riser_3302_2_Zeta3D'].loc[row_id] = riser_3302_2_zeta
            #print('New_Zeta_epsi for riser_LAO_LHS:', target_df['riser_LAO_LHS_Zeta3D'].loc[row_id])
            mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi = fddn_zeta_flag_input(target_df[['HoV','MIXP','AMBP','AMBT']+fddn_input_col_names].loc[[row_id]])
            FDDN_FlowRates_raw_df_epsi = FDDN_Solver(zeta_select_elements, fm_flag_sel_elements, mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi)
            FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
            V_hat_riser_LAO_LHS_epsi_FDDN = FDDN_FlowRates_df_epsi[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']]
            #print('V_hat_riser_LAO_LHS_epsi_FDDN:',V_hat_riser_LAO_LHS_epsi_FDDN.values)

            ## Compute new flow rates for riser_CAO_RHS_cf_epsi
            target_df[MTR+'_Zeta3D'].loc[row_id] = MTR_zeta_tuple[0]
            target_df['riser_LAO_RHS_Zeta3D'].loc[row_id] = riser_LAO_RHS_zeta
            target_df['riser_LAO_LHS_Zeta3D'].loc[row_id] = riser_LAO_LHS_zeta
            target_df['riser_CAO_RHS_Zeta3D'].loc[row_id] = riser_Zeta_3D(riser_CAO_RHS_ZetaCFD, riser_CAO_RHS_cf_epsi)
            target_df['riser_CAO_LHS_Zeta3D'].loc[row_id] = riser_CAO_LHS_zeta
            target_df['riser_3302_2_3305_Zeta3D'].loc[row_id] = riser_3302_2_3305_zeta
            target_df['riser_3302_1_Zeta3D'].loc[row_id] = riser_3302_1_zeta
            target_df['riser_3302_2_Zeta3D'].loc[row_id] = riser_3302_2_zeta
            #print('New_Zeta_epsi for riser_CAO_RHS:', target_df['riser_CAO_RHS_Zeta3D'].loc[row_id])
            mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi = fddn_zeta_flag_input(target_df[['HoV','MIXP','AMBP','AMBT']+fddn_input_col_names].loc[[row_id]])
            FDDN_FlowRates_raw_df_epsi = FDDN_Solver(zeta_select_elements, fm_flag_sel_elements, mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi)
            FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
            V_hat_riser_CAO_RHS_epsi_FDDN = FDDN_FlowRates_df_epsi[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']]
            #print('V_hat_riser_CAO_RHS_epsi_FDDN:',V_hat_riser_CAO_RHS_epsi_FDDN.values)

            ## Compute new flow rates for riser_CAO_LHS_cf_epsi
            target_df[MTR+'_Zeta3D'].loc[row_id] = MTR_zeta_tuple[0]
            target_df['riser_LAO_RHS_Zeta3D'].loc[row_id] = riser_LAO_RHS_zeta
            target_df['riser_LAO_LHS_Zeta3D'].loc[row_id] = riser_LAO_LHS_zeta
            target_df['riser_CAO_RHS_Zeta3D'].loc[row_id] = riser_CAO_RHS_zeta
            target_df['riser_CAO_LHS_Zeta3D'].loc[row_id] = riser_Zeta_3D(riser_CAO_LHS_ZetaCFD, riser_CAO_LHS_cf_epsi)
            target_df['riser_3302_2_3305_Zeta3D'].loc[row_id] = riser_3302_2_3305_zeta
            target_df['riser_3302_1_Zeta3D'].loc[row_id] = riser_3302_1_zeta
            target_df['riser_3302_2_Zeta3D'].loc[row_id] = riser_3302_2_zeta
            #print('New_Zeta_epsi for riser_CAO_LHS:', target_df['riser_CAO_LHS_Zeta3D'].loc[row_id])
            mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi = fddn_zeta_flag_input(target_df[['HoV','MIXP','AMBP','AMBT']+fddn_input_col_names].loc[[row_id]])
            FDDN_FlowRates_raw_df_epsi = FDDN_Solver(zeta_select_elements, fm_flag_sel_elements, mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi)
            FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
            V_hat_riser_CAO_LHS_epsi_FDDN = FDDN_FlowRates_df_epsi[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']]
            #print('V_hat_riser_CAO_LHS_epsi_FDDN:',V_hat_riser_CAO_LHS_epsi_FDDN.values)

            ## Compute new flow rates for riser_3302_2_3305_cf_epsi
            target_df[MTR+'_Zeta3D'].loc[row_id] = MTR_zeta_tuple[0]
            target_df['riser_LAO_RHS_Zeta3D'].loc[row_id] = riser_LAO_RHS_zeta
            target_df['riser_LAO_LHS_Zeta3D'].loc[row_id] = riser_LAO_LHS_zeta
            target_df['riser_CAO_RHS_Zeta3D'].loc[row_id] = riser_CAO_RHS_zeta
            target_df['riser_CAO_LHS_Zeta3D'].loc[row_id] = riser_CAO_LHS_zeta
            target_df['riser_3302_2_3305_Zeta3D'].loc[row_id] = riser_Zeta_3D(riser_3302_2_3305_ZetaCFD, riser_3302_2_3305_cf_epsi)
            target_df['riser_3302_1_Zeta3D'].loc[row_id] = riser_3302_1_zeta
            target_df['riser_3302_2_Zeta3D'].loc[row_id] = riser_3302_2_zeta
            #print('New_Zeta_epsi for riser_CAO_LHS:', target_df['riser_CAO_LHS_Zeta3D'].loc[row_id])
            mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi = fddn_zeta_flag_input(target_df[['HoV','MIXP','AMBP','AMBT']+fddn_input_col_names].loc[[row_id]])
            FDDN_FlowRates_raw_df_epsi = FDDN_Solver(zeta_select_elements, fm_flag_sel_elements, mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi)
            FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
            V_hat_riser_3302_2_3305_epsi_FDDN = FDDN_FlowRates_df_epsi[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']]
            #print('V_hat_riser_3302_1_epsi_FDDN:',V_hat_riser_3302_1_epsi_FDDN.values)

            ## Compute new flow rates for riser_3302_1_cf_epsi
            target_df[MTR+'_Zeta3D'].loc[row_id] = MTR_zeta_tuple[0]
            target_df['riser_LAO_RHS_Zeta3D'].loc[row_id] = riser_LAO_RHS_zeta
            target_df['riser_LAO_LHS_Zeta3D'].loc[row_id] = riser_LAO_LHS_zeta
            target_df['riser_CAO_RHS_Zeta3D'].loc[row_id] = riser_CAO_RHS_zeta
            target_df['riser_CAO_LHS_Zeta3D'].loc[row_id] = riser_CAO_LHS_zeta
            target_df['riser_3302_2_3305_Zeta3D'].loc[row_id] = riser_3302_2_3305_zeta
            target_df['riser_3302_1_Zeta3D'].loc[row_id] = riser_Zeta_3D(riser_3302_1_ZetaCFD, riser_3302_1_cf_epsi)
            target_df['riser_3302_2_Zeta3D'].loc[row_id] = riser_3302_2_zeta
            #print('New_Zeta_epsi for riser_CAO_LHS:', target_df['riser_CAO_LHS_Zeta3D'].loc[row_id])
            mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi = fddn_zeta_flag_input(target_df[['HoV','MIXP','AMBP','AMBT']+fddn_input_col_names].loc[[row_id]])
            FDDN_FlowRates_raw_df_epsi = FDDN_Solver(zeta_select_elements, fm_flag_sel_elements, mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi)
            FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
            V_hat_riser_3302_1_epsi_FDDN = FDDN_FlowRates_df_epsi[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']]
            #print('V_hat_riser_3302_1_epsi_FDDN:',V_hat_riser_3302_1_epsi_FDDN.values)

            ## Compute new flow rates for riser_3302_2_cf_epsi
            target_df[MTR+'_Zeta3D'].loc[row_id] = MTR_zeta_tuple[0]
            target_df['riser_LAO_RHS_Zeta3D'].loc[row_id] = riser_LAO_RHS_zeta
            target_df['riser_LAO_LHS_Zeta3D'].loc[row_id] = riser_LAO_LHS_zeta
            target_df['riser_CAO_RHS_Zeta3D'].loc[row_id] = riser_CAO_RHS_zeta
            target_df['riser_CAO_LHS_Zeta3D'].loc[row_id] = riser_CAO_LHS_zeta
            target_df['riser_3302_2_3305_Zeta3D'].loc[row_id] = riser_3302_2_3305_zeta
            target_df['riser_3302_1_Zeta3D'].loc[row_id] = riser_3302_1_zeta
            target_df['riser_3302_2_Zeta3D'].loc[row_id] = riser_Zeta_3D(riser_3302_2_ZetaCFD, riser_3302_2_cf_epsi)
            #print('New_Zeta_epsi for riser_CAO_LHS:', target_df['riser_CAO_LHS_Zeta3D'].loc[row_id])
            mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi = fddn_zeta_flag_input(target_df[['HoV','MIXP','AMBP','AMBT']+fddn_input_col_names].loc[[row_id]])
            FDDN_FlowRates_raw_df_epsi = FDDN_Solver(zeta_select_elements, fm_flag_sel_elements, mixp_input_epsi, ambp_input_epsi, ambt_input_epsi, zeta_input_epsi, fm_flag_input_epsi, fp2_flag_input_epsi)
            FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
            V_hat_riser_3302_2_epsi_FDDN = FDDN_FlowRates_df_epsi[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']]
            #print('V_hat_riser_3302_2_epsi_FDDN:',V_hat_riser_3302_2_epsi_FDDN.values)

            # Compile all epsi flowrates computed into a tuple
            V_hat_epsi_all_tuple = (V_hat_MTR_epsi_FDDN, V_hat_riser_LAO_RHS_epsi_FDDN, V_hat_riser_LAO_LHS_epsi_FDDN, V_hat_riser_CAO_RHS_epsi_FDDN, V_hat_riser_CAO_LHS_epsi_FDDN, V_hat_riser_3302_2_3305_epsi_FDDN, V_hat_riser_3302_1_epsi_FDDN, V_hat_riser_3302_2_epsi_FDDN)

        V_true_LTR = target_df[['TZ3_Flow','LAORH_SumFlow','LAOLH_SumFlow','CAORH_SumFlow','CAOLH_SumFlow']].loc[[row_id]]
        # print('LTR_TZ3_FlowRate:',V_true_LTR.values)
        # print('FDDN_TZ3_Flowrate:', V_hat_FDDN.values)
        # print('V_hat_epsi_all_tuple:', V_hat_epsi_all_tuple, type(V_hat_epsi_all_tuple))
        return V_true_LTR, V_hat_FDDN, V_hat_epsi_all_tuple


    def backward_prop(self, model, activations, X, c_f, V_max_org, V_true_LTR, V_hat_FDDN, V_hat_epsi_all_tuple):

        m = X.shape[0]     # Get number of samples
        # N = m * 8          # multiple no. of data points with the no. of outputs we are predicting

        # Load V_epsi flowrates as np.array
        V_hat_MTR_epsi_FDDN, V_hat_riser_LAO_RHS_epsi_FDDN, V_hat_riser_LAO_LHS_epsi_FDDN, V_hat_riser_CAO_RHS_epsi_FDDN, V_hat_riser_CAO_LHS_epsi_FDDN, V_hat_riser_3302_2_3305_epsi_FDDN, V_hat_riser_3302_1_epsi_FDDN, V_hat_riser_3302_2_epsi_FDDN = V_hat_epsi_all_tuple

        if (m == 1):
            V_hat_MTR_epsi_FDDN = np.asarray(V_hat_MTR_epsi_FDDN)
            V_hat_riser_LAO_RHS_epsi_FDDN = np.asarray(V_hat_riser_LAO_RHS_epsi_FDDN)
            V_hat_riser_LAO_LHS_epsi_FDDN = np.asarray(V_hat_riser_LAO_LHS_epsi_FDDN)
            V_hat_riser_CAO_RHS_epsi_FDDN = np.asarray(V_hat_riser_CAO_RHS_epsi_FDDN)
            V_hat_riser_CAO_LHS_epsi_FDDN = np.asarray(V_hat_riser_CAO_LHS_epsi_FDDN)
            V_hat_riser_3302_2_3305_epsi_FDDN = np.asarray(V_hat_riser_3302_2_3305_epsi_FDDN)
            V_hat_riser_3302_1_epsi_FDDN = np.asarray(V_hat_riser_3302_1_epsi_FDDN)
            V_hat_riser_3302_2_epsi_FDDN = np.asarray(V_hat_riser_3302_2_epsi_FDDN)
        else:
            V_hat_MTR_epsi_FDDN = V_hat_MTR_epsi_FDDN.values # np.reshape(V_hat_MTR_epsi_FDDN.values,(m,V_hat_MTR_epsi_FDDN.shape[1]))
            V_hat_riser_LAO_RHS_epsi_FDDN = V_hat_riser_LAO_RHS_epsi_FDDN.values
            V_hat_riser_LAO_LHS_epsi_FDDN = V_hat_riser_LAO_LHS_epsi_FDDN.values
            V_hat_riser_CAO_RHS_epsi_FDDN = V_hat_riser_CAO_RHS_epsi_FDDN.values
            V_hat_riser_CAO_LHS_epsi_FDDN = V_hat_riser_CAO_LHS_epsi_FDDN.values
            V_hat_riser_3302_2_3305_epsi_FDDN = V_hat_riser_3302_2_3305_epsi_FDDN.values
            V_hat_riser_3302_1_epsi_FDDN = V_hat_riser_3302_1_epsi_FDDN.values
            V_hat_riser_3302_2_epsi_FDDN = V_hat_riser_3302_2_epsi_FDDN.values

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

        V_max = V_max_org.values # np.append((V_max_org.values),[1])

        # print('\nBACKWARD PROP')
        # print('m:',m)
        # print('Input Features:',list(X), type(X), X.shape)
        # print('cf_array:',list(c_f), type(c_f), c_f.shape)
        # print('V_max:',list(V_max), type(V_max), V_max.shape)
        # print('V_true_LTR:',list(V_true_LTR.values), type(V_true_LTR), V_true_LTR.shape)
        # print('V_hat_FDDN:',list(V_hat_FDDN.values), type(V_hat_FDDN), V_hat_FDDN.shape)
        # print('V_hat_MTR_epsi_FDDN:', list(V_hat_MTR_epsi_FDDN), type(V_hat_MTR_epsi_FDDN), V_hat_MTR_epsi_FDDN.shape )
        # print('V_hat_riser_CAO_LHS_epsi_FDDN:', list(V_hat_riser_CAO_LHS_epsi_FDDN), type(V_hat_riser_CAO_LHS_epsi_FDDN), V_hat_riser_CAO_LHS_epsi_FDDN.shape)

        # Derivative of flowrate with respect to neural network o/p (i.e correction factors)
        dv_da_MTR_epsi           = (V_hat_MTR_epsi_FDDN            - V_hat_FDDN.values) / self.MTR_epsi
        dv_da_riser_LAO_RHS_epsi = (V_hat_riser_LAO_RHS_epsi_FDDN  - V_hat_FDDN.values) / self.riser_LAO_RHS_epsi
        dv_da_riser_LAO_LHS_epsi = (V_hat_riser_LAO_LHS_epsi_FDDN  - V_hat_FDDN.values) / self.riser_LAO_LHS_epsi
        dv_da_riser_CAO_RHS_epsi = (V_hat_riser_CAO_RHS_epsi_FDDN  - V_hat_FDDN.values) / self.riser_CAO_RHS_epsi
        dv_da_riser_CAO_LHS_epsi = (V_hat_riser_CAO_LHS_epsi_FDDN  - V_hat_FDDN.values) / self.riser_CAO_LHS_epsi
        dv_da_riser_3302_2_3305_epsi = (V_hat_riser_3302_2_3305_epsi_FDDN  - V_hat_FDDN.values) / self.riser_3302_2_3305_epsi
        dv_da_riser_3302_1_epsi = (V_hat_riser_3302_1_epsi_FDDN  - V_hat_FDDN.values) / self.riser_3302_1_epsi
        dv_da_riser_3302_2_epsi = (V_hat_riser_3302_2_epsi_FDDN  - V_hat_FDDN.values) / self.riser_3302_2_epsi

        # print('epsi:',self.MTR_epsi,self.riser_LAO_RHS_epsi,self.riser_LAO_LHS_epsi,self.riser_CAO_RHS_epsi,self.riser_CAO_LHS_epsi,self.riser_3302_1_epsi)
        # print('dv_da_MTR_epsi:',list(dv_da_MTR_epsi),type(dv_da_MTR_epsi),'Shape:', dv_da_MTR_epsi.shape)
        # print('dv_da_riser_LAO_RHS_epsi:',list(dv_da_riser_LAO_RHS_epsi), type(dv_da_riser_LAO_RHS_epsi), dv_da_riser_LAO_RHS_epsi.shape)
        # print('dv_da_riser_LAO_LHS_epsi:',list(dv_da_riser_LAO_LHS_epsi), type(dv_da_riser_LAO_LHS_epsi), dv_da_riser_LAO_LHS_epsi.shape)
        # print('dv_da_riser_CAO_RHS_epsi:',list(dv_da_riser_CAO_RHS_epsi), type(dv_da_riser_CAO_RHS_epsi), dv_da_riser_CAO_RHS_epsi.shape)
        # print('dv_da_riser_CAO_LHS_epsi:',list(dv_da_riser_CAO_LHS_epsi), type(dv_da_riser_CAO_LHS_epsi), dv_da_riser_CAO_LHS_epsi.shape)
        # print('dv_da_riser_3302_2_3305_epsi:',list(dv_da_riser_3302_2_3305_epsi), dv_da_riser_3302_2_3305_epsi.shape)
        # print('dv_da_riser_3302_1_epsi:',list(dv_da_riser_3302_1_epsi), dv_da_riser_3302_1_epsi.shape)
        # print('dv_da_riser_3302_2_epsi:',list(dv_da_riser_3302_2_epsi), dv_da_riser_3302_1_epsi.shape)
        dv_da = np.vstack((dv_da_MTR_epsi.ravel(), dv_da_riser_LAO_RHS_epsi.ravel(), dv_da_riser_LAO_LHS_epsi.ravel(), dv_da_riser_CAO_RHS_epsi.ravel(), dv_da_riser_CAO_LHS_epsi.ravel(), dv_da_riser_3302_2_3305_epsi.ravel(), dv_da_riser_3302_1_epsi.ravel(), dv_da_riser_3302_2_epsi.ravel()))
        # print('\ndv_da:',list(dv_da),type(dv_da),'dv_da Shape:', dv_da.shape)

        flowrate_diff = V_hat_FDDN.values - V_true_LTR.values
        print('Flowrate_delta (FDDN - LTR):', list(flowrate_diff), type(flowrate_diff), flowrate_diff.shape)

        # Backpropagation
        # # Correction_factor output from final output layer having linear (identity) activation function, so g'(z_k) = 1
        if (self.error_loss_choice == 'simplified'):  # Calculate error
            error =   self.cf_lambda_reg_factor * (1/(2 * m)) * ( 1/V_max**2 * flowrate_diff**2).sum()
            delta_k0 = self.cf_lambda_reg_factor * (1/m)  * np.dot((1/V_max**2 * flowrate_diff).ravel(),dv_da.T) + np.zeros_like(c_f) #Add with empty array of zeros to match delta_k shape
        elif (self.error_loss_choice == 'cf_only'):
            error = (1/(2 * m)) * ((c_f - 1)**2).sum() # Calculate error
            delta_k0 = (1/m) * (c_f - 1)
        elif (self.error_loss_choice == 'full'):
            error = self.cf_lambda_reg_factor * (1/(2 * m)) * (( 1/V_max**2 * flowrate_diff**2).sum() + (1/self.cf_lambda_reg_factor) * ((c_f - 1)**2).sum() )  # Calculate error
            delta_k0 = self.cf_lambda_reg_factor * (1/m) *  (np.dot((1/V_max**2 * flowrate_diff).ravel(),dv_da.T)  + (1/self.cf_lambda_reg_factor) * (c_f - 1) ) # Delta-k - dE/dWjk
        elif (self.error_loss_choice == 'hyperbola_regularization'):
            error = self.cf_lambda_reg_factor * (1/(2 * m)) * (( 1/V_max**2 * flowrate_diff**2).sum() + (1/self.cf_lambda_reg_factor) * (( np.maximum(c_f, 1 /c_f) - 1)**2).sum()) # Calculate error
            delta_k0 = self.cf_lambda_reg_factor * (1/m) * ( np.dot((1/V_max**2 * flowrate_diff).ravel(),dv_da.T)  + (1/self.cf_lambda_reg_factor) * np.maximum((c_f - 1),((1 - 1 /c_f)) * (1 /c_f)**2))  # Delta-k - dE/dWjk

        dz_dict[dz_dict_keys[-1]] = delta_k0
        # print("\nError:",error, type(error), "Shape:",error.shape)
        # print("Delta_k ({}):".format(dz_dict_keys[-1]),list(dz_dict[dz_dict_keys[-1]]), "Shape:",dz_dict[dz_dict_keys[-1]].shape)
        dW_dict[dW_dict_keys[-1]] =  (1/m)*np.dot(np.asarray(activations[activations_keys[-2]]).T, dz_dict[dz_dict_keys[-1]])
        # print("{}:".format(dW_dict_keys[-1]),list(dW_dict[dW_dict_keys[-1]]), "Shape:",dW_dict[dW_dict_keys[-1]].shape)
        db_dict[db_dict_keys[-1]] = (1/m)*np.sum(dz_dict[dz_dict_keys[-1]], axis=0)
        # print("{}:".format(db_dict_keys[-1]),list(db_dict[db_dict_keys[-1]]), "Shape:",db_dict[db_dict_keys[-1]].shape)
        for w_i, k in zip(range(1,len(weight_keys)), range(2,len(activations)+1)):
            # print(weight_keys[-(w_i)],'Shape:',  model[weight_keys[-(w_i)]].shape) #list(model[weight_keys[-(w_i)]]),
            # print(activations_keys[-k],'Length:', len(activations[activations_keys[-k]]))
            dz_dict[dz_dict_keys[-(k)]] = np.multiply(np.dot(dz_dict[dz_dict_keys[-(w_i)]], model[weight_keys[-(w_i)]].T), self.activation_function_derivative(np.asarray(activations[activations_keys[-(k)]])))
            # print("{}:".format(dz_dict_keys[-k]),list(dz_dict[dz_dict_keys[-k]]), "Shape:",dz_dict[dz_dict_keys[-k]].shape)
            dW_dict[dW_dict_keys[-(k)]] =  (1/m)*np.dot(np.asarray(activations[activations_keys[-(k+1)]]).T, dz_dict[dz_dict_keys[-(k)]])
            # print("{}:".format(dW_dict_keys[-k]),list(dW_dict[dW_dict_keys[-k]]), "Shape:",dW_dict[dW_dict_keys[-k]].shape)
            db_dict[db_dict_keys[-(k)]] =  (1/m)* np.sum(dz_dict[dz_dict_keys[-(k)]], axis=0)
            # print("{}:".format(db_dict_keys[-k]),list(db_dict[db_dict_keys[-k]]), "Shape:",db_dict[db_dict_keys[-k]].shape)
        return error, dW_dict, db_dict


    def update_parameters(self, lr, model_old, weight_gradients, bias_gradients):
        # print('UPDATE PARAMETERS - CURRENT WEIGHT and BIAS GRADIENTS:')
        # pprint_dict(weight_gradients),pprint_dict(bias_gradients)
        # Load model parameters
        weights_keys = [k for k, v in model_old.items() if k.startswith('W')]
        biases_keys = [k for k, v in model_old.items() if k.startswith('b')]
        weight_gradients_keys = list(weight_gradients.keys())
        bias_gradients_keys = list(bias_gradients.keys())
        model_new = deepcopy(model_old)
        # Update parameters
        for w_k, dw_k, b_k, db_k in zip(weights_keys, weight_gradients_keys, biases_keys, bias_gradients_keys ):
            model_new[w_k] -= lr * weight_gradients[dw_k]
            model_new[b_k] -= lr * bias_gradients[db_k]
        # print('MODEL WEIGHTS & BIASES UPDATED:\n', model_new)
        return model_new

    def predict(self, model, X, row_id, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df):
        # Do forward pass
        _, org_cf_tuple , _ , _ = self.forward_prop(model, X)
        epsi_cf_tuple = tuple(None for cf in org_cf_tuple) # Create a tuple of Nones for the same length as org_cf_tuple
        # FDDN FlowRates
        V_true_LTR, V_hat_FDDN, _  = self.FDDN_SolverProcess(org_cf_tuple, epsi_cf_tuple, row_id, restrictor_params, restrictor_duct_areas, fddn_input_col_names, target_df)
        delta_V = pd.DataFrame(columns = ['Delta_TZ3_Flow','Delta_LAORH_SumFlow','Delta_LAOLH_SumFlow','Delta_CAORH_SumFlow','Delta_CAOLH_SumFlow'])
        delta_V['Delta_TZ3_Flow'] = V_hat_FDDN['TZ3_Flow'].values - V_true_LTR['TZ3_Flow'].values
        delta_V['Delta_LAORH_SumFlow'] = V_hat_FDDN['LAORH_SumFlow'].values - V_true_LTR['LAORH_SumFlow'].values
        delta_V['Delta_LAOLH_SumFlow'] = V_hat_FDDN['LAOLH_SumFlow'].values - V_true_LTR['LAOLH_SumFlow'].values
        delta_V['Delta_CAORH_SumFlow'] = V_hat_FDDN['CAORH_SumFlow'].values - V_true_LTR['CAORH_SumFlow'].values
        delta_V['Delta_CAOLH_SumFlow'] = V_hat_FDDN['CAOLH_SumFlow'].values - V_true_LTR['CAOLH_SumFlow'].values
        print("Predicted CFs:", org_cf_tuple)
        return org_cf_tuple, V_true_LTR, V_hat_FDDN, delta_V
