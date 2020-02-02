## This script contains the Hybrid PINN class with adaptive learning Rate computed using Line Search.
## Author: Dilip Rajkumar
## Version Description: In this version (v2.1) the neural network outputs 1 correction factor and uses minibatch gradient descent with adaptive batch size
## Linear activation function is only used.

import pandas as pd
import numpy as np
from FDDN_Lib import fddn_zeta_input,FDDN_Solver, FDDN_output_df_gen, zeta_values

MTR = 'R600_HD'
select_elements = ['MTR-600','R-610', 'R-611', 'R-612', 'R-613','CAOR-620_fwd', 'CAOR-620_mid', 'CAOR-620_aft', 'CAOR-621_fwd', 'CAOR-621_mid', 'CAOR-621_aft']

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

# Define Custom Class for Physics Informed Neural Network
class PINN(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, epsi):
        # Set number of nodes (neurons) in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.epsi = epsi

        ## Custom weights initialization: w_ij = [1, 0.1] and w_jk = [1]
        self.weights_input_to_hidden     =  np.ones((self.input_nodes, self.hidden_nodes))
        self.weights_input_to_hidden[1]  = 0.1
        self.weights_hidden_to_output = np.ones((self.hidden_nodes, self.output_nodes))
        print('Initial Weights_i_h:', list(self.weights_input_to_hidden))
        print('Initial Weights_h_o:', list(self.weights_hidden_to_output))

        ## New calc weights
        self.new_weights_i_h = np.zeros((self.input_nodes, self.hidden_nodes))
        self.new_weights_h_o = np.zeros((self.hidden_nodes, self.output_nodes))


    def train(self, features, diameters, batch_idx, w_ih, w_ho, MTR_DuctArea, target_df, zeta_col_names):
        ''' Train the network on batch of input features.

            Arguments
            ---------
            features:  2D array of input features, each row is one data record, each column is a feature
            diameters: diameter array input for calculating Zeta using the `SHR_Zeta_3D` function
            batch_idx: indices of the training data in the current batch
            MTR_DuctArea:  Duct Area for the Mixer Tapping Restrictor
            target_df:     Source dataframe containing required columns
            zeta_col_names:Zeta names of all the elements

        '''
        n_records = features.shape[0]
        ## Assign default initial weights if weight variables are set to None
        if (w_ih is None):
            w_ih = self.weights_input_to_hidden

        if (w_ho is None):
            w_ho = self.weights_hidden_to_output

        V_true_LTR_list, V_hat_FDDN_list, V_hat_epsi_FDDN_list, hidden_output_list, break_flag_arr, cf_list  = [], [], [], [], [], []
        # Run forward pass and FDDN_SolverProcess for each data point in the current batch
        for X, row_id, dia in zip(features, batch_idx, diameters):
            c_f, cf_epsi, final_outputs, hidden_outputs = self.forward_pass(X, w_ih, w_ho)  # Implement the forward pass
            V_true_LTR,V_hat_FDDN,V_hat_epsi_FDDN = self.FDDN_SolverProcess(c_f, cf_epsi, row_id, dia, MTR_DuctArea, zeta_col_names,target_df)
            ## Calculate Percentage Difference between flowrates for early stopping
            percentage_diff = 100 * 2 * (abs(V_true_LTR - V_hat_FDDN)) / (V_true_LTR + V_hat_FDDN)
            if (percentage_diff < 0.3): # % diff < 0.3%
                break_flag = 1
                print('% Diff between flowrates',percentage_diff[0],'%')
            else:
                break_flag = 0

            # Append elements into a list
            break_flag_arr.append(break_flag)
            V_true_LTR_list.append(V_true_LTR[0])
            V_hat_FDDN_list.append(V_hat_FDDN[0])
            V_hat_epsi_FDDN_list.append(V_hat_epsi_FDDN[0])
            hidden_output_list.append(hidden_outputs)
            cf_list.append(c_f[0])

        # Convert Lists to Numpy arrays and reshape them according to the batch size
        hidden_output_array = np.reshape(np.asarray(hidden_output_list), (self.hidden_nodes, n_records))
        V_true_LTR_arr = np.reshape(np.asarray(V_true_LTR_list), (n_records,1))
        V_hat_FDDN_arr = np.reshape(np.asarray(V_hat_FDDN_list), (n_records,1))
        V_hat_epsi_FDDN_arr = np.reshape(np.asarray(V_hat_epsi_FDDN_list), (n_records, 1))
        cf_array = np.asarray(cf_list)

        return V_true_LTR_arr, V_hat_FDDN_arr, V_hat_epsi_FDDN_arr, cf_array, hidden_output_array, break_flag_arr

    def forward_pass(self, X, w_ih, w_ho):
        ### FORWARD pass ###
        hidden_inputs = np.dot(X, w_ih) # signals into hidden layer
        hidden_outputs = hidden_inputs # Linear activation function
        # Output layer
        final_inputs = np.dot(hidden_outputs, w_ho) # signals into final output layer
        c_f = final_inputs # Correction_factor from final output layer having linear (identity) activation function

        if (c_f[0] < 0):
            cf_epsi = c_f - self.epsi
            c_f = abs(c_f)
            print('\nNN output is -ve. |(c_f)| value:',c_f)
        elif (c_f[0] > 0):
            print('\nNN output (C_f):',c_f[0])
            cf_epsi = c_f + self.epsi

        return c_f, cf_epsi, final_inputs, hidden_outputs

    def FDDN_SolverProcess(self, c_f, cf_epsi, row_id, dia, MTR_DuctArea, zeta_col_names,target_df):
        ## Pass Neural network output (correction factor) to compute new Zeta and then re-run FDDN solver
        _, zeta_tuple = zip(*[SHR_Zeta_3D(1,dia,MTR_DuctArea,c_f)])
        target_df[MTR+'_Zeta3D'].loc[row_id] = zeta_tuple[0] # Assign first element in zeta_tuple to required index
        print('New_Zeta:',target_df[MTR+'_Zeta3D'].loc[row_id])

        ## Call FDDN Solver to compute new flowrates with the updated Zeta
        mixp_input,ambp_input,ambt_input,zeta_input = fddn_zeta_input(target_df[['HoV','MIXP','AMBP','AMBT','TZ6_Flow']+zeta_col_names].loc[[row_id]])
        FDDN_FlowRates_raw_df = FDDN_Solver(select_elements,mixp_input,ambp_input,ambt_input,zeta_input)
        FDDN_FlowRates_df = FDDN_output_df_gen(FDDN_FlowRates_raw_df )
        V_hat_FDDN = FDDN_FlowRates_df['TZ6_Flow'].values/1000

        ## Pass Neural network output (correction factor) to compute new Zeta and then re-run FDDN solver
        _, zeta_tuple_epsi = zip(*[SHR_Zeta_3D(1,dia,MTR_DuctArea,cf_epsi)])
        target_df[MTR+'_Zeta3D'].loc[row_id] = zeta_tuple_epsi[0] # Assign first element in zeta_tuple to required index
        print('New_Zeta_epsi:',target_df[MTR+'_Zeta3D'].loc[row_id])

         ## Call FDDN Solver to compute new flowrates with the updated Zeta with Epsilon
        mixp_input_epsi,ambp_input_epsi,ambt_input_epsi,zeta_input_epsi = fddn_zeta_input(target_df[['HoV','MIXP','AMBP','AMBT','TZ6_Flow']+zeta_col_names].loc[[row_id]])
        FDDN_FlowRates_raw_df_epsi = FDDN_Solver(select_elements,mixp_input_epsi,ambp_input_epsi,ambt_input_epsi,zeta_input_epsi)
        FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
        V_hat_epsi_FDDN = FDDN_FlowRates_df_epsi['TZ6_Flow'].values/1000
        V_true_LTR = target_df['TZ6_Flow'].loc[[row_id]].values

        print('Row ID:',row_id)
        print('HoV:', target_df['HoV'].loc[[row_id]].values)
        # print('LTR_TZ6_FlowRate:',V_true_LTR)
        # print('FDDN_TZ6_Flowrate:', V_hat_FDDN)
        # print('FDDN_TZ6_Flowrate_epsi:', V_hat_epsi_FDDN)
        print('Flow Rate Difference (LTR - FDDN):', V_true_LTR[0] - V_hat_FDDN[0], 'cu.m/s')
        return V_true_LTR,V_hat_FDDN,V_hat_epsi_FDDN

    def gradient_update(self, V_true_LTR, V_hat_FDDN, V_hat_epsi_FDDN, c_f, X, hidden_outputs, delta_weights_i_h, delta_weights_h_o, w_ho, V_max):
        n_records = X.shape[0] # Because we pass n data points in every batch
        print('Nr. of Records:',X)
        if (delta_weights_i_h is None) & (delta_weights_h_o is None):
            dw_ih = np.zeros(self.weights_input_to_hidden.shape) # Create empty array filled with zeros
            dw_ho = np.zeros(self.weights_hidden_to_output.shape) # Create empty array filled with zeros
        else:
            dw_ih = np.copy(delta_weights_i_h)
            dw_ho = np.copy(delta_weights_h_o)

        if (w_ho is None):
            w_ho = self.weights_hidden_to_output
        else:
            w_ho = np.reshape(w_ho,(self.weights_hidden_to_output.shape))

        # Compute Error function
        error = (1/(2 * (V_max)**2)) * ((V_hat_FDDN - V_true_LTR)**2).sum()  # + (1/2) * ((c_f_array - 1)**2).sum() # Output error

        ### BACKWARD pass - Backpropagated error terms ###
        # Derivative of flowrate with respect to neural network o/p (i.e correction factor)
        dv_da = (V_hat_epsi_FDDN - V_hat_FDDN) / self.epsi

        output_error_term = (1/(V_max)**2) * (V_hat_FDDN - V_true_LTR)*dv_da # + (1/n_records) * (c_f - 1) # Delta_k

        # Weight step (hidden to output) - # REMOVE += in below line when using the np.dot
        dw_ho += 0 # np.dot(hidden_outputs, output_error_term.T) ## DELTA Weights , i.e dE/dwjk

        # Hidden layer's contribution to the error#
        hidden_error = np.dot(output_error_term, w_ho)

        hidden_error_term = hidden_error * 1 # Delta_j For Linear activation return it's derivative 1

        # # Weight step (input to hidden)
        if (n_records == 1): #  singleHoV loop all HoVs converged in < 7 mins
            dw_ih = np.dot(X.T, hidden_error_term)
        else: # if the below order for dw_ih is used as default it causes poor performance for minibatch and does not work for SingleHoV Loop
            dw_ih = np.dot(X, hidden_error_term) #i.e dE/dwij -

        print('GRADIENT_UPDATE - dw_ih:', list(dw_ih), type(dw_ih), dw_ih.shape)
        print('GRADIENT_UPDATE - dw_ho', list(dw_ho), type(dw_ho), dw_ho.shape)
        return error, dw_ih, dw_ho

    def cust_f_alpha(self, dw_i_h, dw_h_o, dw_i_h_new, dw_h_o_new):
        e0 = np.concatenate((dw_i_h.flatten(),     dw_h_o.flatten()), axis=0)
        e1 = np.concatenate((dw_i_h_new.flatten(), dw_h_o_new.flatten()), axis=0)
        # print('\nCombined Delta_Weights_iho_Old (E0):',e0)
        # print('Magnitude of e0:',mag_vector(e0))
        # print('Combined Delta_Weights_iho_New (E1):',e1)
        # print('Magnitude of e1:',mag_vector(e1))
        return np.dot(e0, e1)

    def calc_lr2(self, f0,f1,lr0,lr1):
        lr2 = -(f0) / ((f1 - f0)/(lr1 - lr0))
        print('Learning Rate 2:', lr2)

        # Set min and max threshold limits for Learning rate 2
        if (lr2 < 0):
            lr2 = 0.01 # if lr2 calculated is -ve set it to 0.01
        elif (lr2 > 10):
            lr2 = 10 # if lr2 calculated is exceeds +10 set it to 10
        return lr2

    def calc_new_weights(self, lr, delta_weights_i_h, delta_weights_h_o, w_ih = None, w_ho = None):
        # Update the weights
        print('\nCALC NEW WEIGHTS - Learning Rate Used:',lr)

        if (delta_weights_i_h is None) & (delta_weights_h_o is None):
            dw_ih = np.zeros(self.weights_input_to_hidden.shape) # Create empty array filled with zeros
            dw_ho = np.zeros(self.weights_hidden_to_output.shape) # Create empty array filled with zeros
        else:
            dw_ih = np.copy(delta_weights_i_h)
            dw_ho = np.copy(delta_weights_h_o)
        print('CALC NEW WEIGHTS - delta_weights_input_i_h_o:',list(dw_ih)+list(dw_ho))

        ## Assign default initial weights is weight variables are set to None
        if (w_ih is None):
            w_ih = self.weights_input_to_hidden
        if (w_ho is None):
            w_ho = self.weights_hidden_to_output
        else:
            w_ih = np.reshape(w_ih,(self.weights_input_to_hidden.shape))
            print('CALC NEW WEIGHTS - Inputs Passed Weights_i_h:',list(w_ih),'Shape:',w_ih.shape)
            w_ho = np.reshape(w_ho,(self.weights_hidden_to_output.shape))
            print('CALC NEW WEIGHTS - Inputs Passed Weights_h_o:',list(w_ho),'Shape:',w_ho.shape)

        self.new_weights_i_h  = w_ih - lr * dw_ih # update input-to-hidden weights with gradient descent step
        self.new_weights_h_o  = w_ho - lr * dw_ho # update hidden-to-output weights with gradient descent step
        print('CALC NEW WEIGHTS - new_weights_i_h:',list(self.new_weights_i_h))
        print('CALC NEW WEIGHTS - new_weights_h_o:',list(self.new_weights_h_o))
        return self.new_weights_i_h, self.new_weights_h_o


    def predict(self, features, dia, row_id,w_ih,w_ho):
        ''' Run a forward pass through the network with input features to make predictions.

            Arguments
            ---------
            features: 1D array of feature values
        '''
        c_f,_,_,_ = self.forward_pass(features,w_ih,w_ho)
        return c_f
