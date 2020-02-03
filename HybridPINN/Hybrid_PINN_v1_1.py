## This script contains the Hybrid PINN class structured for minibatch training for the General Approach
## Author: Dilip Rajkumar
## Version Description: In this version (1.1) the neural network outputs 1 correction factor with adaptive batch size.
## Also has a early_stopping function to stop training if error does not improve after "n" epochs.



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
    '''Normalizes each column(feature) in a dataframe by the max value in that column'''
    df = pd.DataFrame()
    for e in features:
        max_val = max_val_dict[e]
        df[e] = df_scaled[e] * max_val
    return df[features]

# Define Custom Class for Physics Informed Neural Network
class PINN(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, epsi):
        # Set number of nodes (neurons) in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        self.epsi = epsi

        # Initialize weights with random values from a random normal distribution
        self.weights_input_to_hidden  = np.random.normal(0.0, self.input_nodes**-0.5,(self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,(self.hidden_nodes, self.output_nodes))

        #### Set self.activation_function to the choice of function
        def relu(x):
            return x * (x > 0)

        self.activation_function = relu


    def train(self, features, diameters, batch_idx , MTR_DuctArea, target_df, zeta_col_names,V_max_org):
        ''' Train the network on batch of input features.

            Arguments
            ---------
            features:  2D array of input features, each row is one data record, each column is a feature
            diameters: diameter array input for calculating Zeta using the `SHR_Zeta_3D` function
            batch_idx: indices of the training data in the current batch
            MTR_DuctArea:  Duct Area for the Mixer Tapping Restrictor
            target_df:     Source dataframe containing required columns
            zeta_col_names:Zeta names of all the elements
            V_max_org:     Max Value of Flow Rate column

        '''
        n_records = features.shape[0]

        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)  # Create empty array filled with zeros
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape) # Create empty array filled with zeros
        V_true_LTR_arr, V_hat_FDDN_arr, V_hat_epsi_FDDN_arr, hidden_output_array, break_flag_arr, c_f_array  = [], [], [], [], [], []
        # Run forward pass and FDDN_SolverProcess for each data point in the current batch
        for X, row_id, dia in zip(features, batch_idx, diameters):
            c_f, cf_epsi, final_outputs, hidden_outputs = self.forward_pass(X)  # Implement the forward pass
            V_true_LTR,V_hat_FDDN,V_hat_epsi_FDDN = self.FDDN_SolverProcess(c_f, cf_epsi, row_id, dia, MTR_DuctArea, zeta_col_names,target_df)
            ## Calculate Percentage Difference between flowrates for early stopping
            percentage_diff = 100 * 2 * (abs(V_true_LTR - V_hat_FDDN)) / (V_true_LTR + V_hat_FDDN)
            if (percentage_diff < 0.3): # % diff < 0.3%
                break_flag = 1
                print('Percentage Diff between flowrates',percentage_diff[0],'%')
            else:
                break_flag = 0

            # Append elements into a list
            break_flag_arr.append(break_flag)
            V_true_LTR_arr.append(V_true_LTR[0])
            V_hat_FDDN_arr.append(V_hat_FDDN[0])
            V_hat_epsi_FDDN_arr.append(V_hat_epsi_FDDN[0])
            hidden_output_array.append(hidden_outputs)
            c_f_array.append(c_f[0])
        error,delta_weights_i_h, delta_weights_h_o = self.backward_pass(features, np.reshape(np.asarray(hidden_output_array), (self.hidden_nodes, n_records)), delta_weights_i_h, delta_weights_h_o,
                                                                  np.reshape(np.asarray(V_hat_FDDN_arr), (1, n_records)), np.reshape(np.asarray(V_hat_epsi_FDDN_arr), (1, n_records)), np.asarray(c_f_array),V_max_org)

        # Update the weights
        self.weights_hidden_to_output += -self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden  += -self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step


        return error,V_hat_FDDN_arr,V_true_LTR_arr,break_flag_arr,c_f_array

    def early_stopping(self,train_loss, last_n_epochs = 10):
        list_elements_diff = np.diff(train_loss[-last_n_epochs:])
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

        if (loss_mag < 0.005):
            print('NO CHANGE in TRAIN LOSS for the Last {} Epochs'.format(last_n_epochs))
            es_bf = 1
        else:
            es_bf = 0

        return es_bf

    def FDDN_SolverProcess(self, c_f, cf_epsi, row_id, dia, MTR_DuctArea, zeta_col_names,target_df):
        ## Pass Neural network output (correction factor) to compute new Zeta and then re-run FDDN solver
        _, zeta_tuple = zip(*[SHR_Zeta_3D(1,dia,MTR_DuctArea,c_f)])
        target_df[MTR+'_Zeta3D'].loc[row_id] = zeta_tuple[0] # Assign first element in zeta_tuple to required index
        print('New_Zeta:',target_df[MTR+'_Zeta3D'].loc[row_id])

        ## Call FDDN Solver to compute new flowrates with the updated Zeta
        mixp_input,ambp_input,ambt_input,zeta_input = fddn_zeta_input(target_df[['HoV','MIXP','AMBP','AMBT','TZ6_Flow']+zeta_col_names].loc[[row_id]])
        FDDN_FlowRates_raw_df = FDDN_Solver(select_elements,mixp_input,ambp_input,ambt_input,zeta_input)
        FDDN_FlowRates_df = FDDN_output_df_gen(FDDN_FlowRates_raw_df )
        V_hat_FDDN = FDDN_FlowRates_df['TZ6_Flow'].values

        ## Pass Neural network output (correction factor) to compute new Zeta and then re-run FDDN solver
        _, zeta_tuple_epsi = zip(*[SHR_Zeta_3D(1,dia,MTR_DuctArea,cf_epsi)])
        target_df[MTR+'_Zeta3D'].loc[row_id] = zeta_tuple_epsi[0] # Assign first element in zeta_tuple to required index
        print('New_Zeta_epsi:',target_df[MTR+'_Zeta3D'].loc[row_id])

         ## Call FDDN Solver to compute new flowrates with the updated Zeta with Epsilon
        mixp_input_epsi,ambp_input_epsi,ambt_input_epsi,zeta_input_epsi = fddn_zeta_input(target_df[['HoV','MIXP','AMBP','AMBT','TZ6_Flow']+zeta_col_names].loc[[row_id]])
        FDDN_FlowRates_raw_df_epsi = FDDN_Solver(select_elements,mixp_input_epsi,ambp_input_epsi,ambt_input_epsi,zeta_input_epsi)
        FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
        V_hat_epsi_FDDN = FDDN_FlowRates_df_epsi['TZ6_Flow'].values
        V_true_LTR = target_df['TZ6_Flow'].loc[[row_id]].values

        print('Row ID:',row_id)
        print('HoV:', target_df['HoV'].loc[[row_id]].values)
        # print('LTR_TZ6_FlowRate:',V_true_LTR)
        # print('FDDN_TZ6_Flowrate:', V_hat_FDDN)
        print('Flow Rate Difference (LTR - FDDN):', V_true_LTR[0] - V_hat_FDDN[0], 'l/s')

        return V_true_LTR,V_hat_FDDN,V_hat_epsi_FDDN

    def backward_pass(self, X, hidden_outputs, delta_weights_i_h, delta_weights_h_o, V_hat_FDDN, V_hat_epsi_FDDN, c_f, V_max_org):
        n_records = X.shape[0]
        print('No. of Records:', n_records)
        V_hat_epsi_FDDN_scaled = V_hat_epsi_FDDN/V_max_org
        V_hat_FDDN_scaled = V_hat_FDDN/V_max_org
        V_max = 1.0 # as the data has already been normalized by max

        ### BACKWARD PASS ###
        # Difference between the FDDN flowrate and LTR Flow Rate and X[:,-1] - Returns the last column from the feature dataset
        flowrate_diff = V_hat_FDDN_scaled.flatten('F') - X[:,-1]
        # print('flowrate_diff:', list(flowrate_diff), type(flowrate_diff), flowrate_diff.shape)
        # Output error
        error = (1/(2 * (V_max)**2)) * ((flowrate_diff)**2).sum() + (1/(2* n_records)) * ((c_f - 1)**2).sum()

        # Derivative of flowrate with respect to neural network o/p (i.e correction factor)
        dv_da = (V_hat_epsi_FDDN - V_hat_FDDN) / self.epsi
        # print('dv_da:',list(dv_da), type(dv_da), dv_da.shape)
        # Backpropagated error terms
        output_error_term = (1/(V_max)**2) * (flowrate_diff)*dv_da + (1/n_records) * (c_f - 1) #Delta_k
        # print('output_error_term:', list(output_error_term), type(output_error_term), output_error_term.shape)

        # Hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        # print('hidden_error:', list(hidden_error), type(hidden_error), hidden_error.shape)
        hidden_error_term = hidden_error * 1 * (hidden_outputs > 0) # For ReLu activation return it's derivative 1.*(h > 0)
        # print('hidden_error_term:', list(hidden_error_term), type(hidden_error_term), hidden_error_term.shape)
        # Weight step (input to hidden)
        delta_weights_i_h += np.dot(X.T, hidden_error_term.T)

        # Weight step (hidden to output)
        delta_weights_h_o += np.dot(hidden_outputs, output_error_term.T)
        # print('delta_weights_i_h:', list(delta_weights_i_h), type(delta_weights_i_h), delta_weights_i_h.shape)
        # print('delta_weights_h_o:', list(delta_weights_h_o), type(delta_weights_h_o), delta_weights_h_o.shape)
        return error, delta_weights_i_h, delta_weights_h_o


    def forward_pass(self, X):
        ### FORWARD pass ###
        # Hidden Layer
        hidden_inputs  = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        c_f = final_inputs # Correction_factor from final output layer having linear (identity) activation function

        if (c_f[0] < 0):
            cf_epsi = c_f - self.epsi
            c_f = abs(c_f)
            print('\nNN output is -ve. Taking |(c_f)| value:',c_f)
        elif (c_f[0] > 0):
            print('\nNN output (C_f):',c_f[0])
            cf_epsi = c_f + self.epsi

        return c_f, cf_epsi, final_inputs, hidden_outputs

    def predict(self, features, dia, row_id):
        ''' Run a forward pass through the network with input features to make predictions.

            Arguments
            ---------
            features: 1D array of feature values
        '''
        c_f,_,_,_ = self.forward_pass(features)
        return c_f
