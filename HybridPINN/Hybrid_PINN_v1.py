## This script contains the class PINN for the Hybrid PINN and some custom scaling functions
## Version Description: In this version (v1) the neural network outputs only one correction factor
## Author: Dilip Rajkumar

import pandas as pd
import numpy as np
from FDDN_Lib import fddn_zeta_input,FDDN_Solver, FDDN_output_df_gen, zeta_values

MTR = 'R600_HD'

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

# Define Custom Classes for Neural Network
class PINN(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes (neurons) in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        # Initialize weights with random values from a random normal distribution
        self.weights_input_to_hidden  = np.random.normal(0.0, self.input_nodes**-0.5,(self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,(self.hidden_nodes, self.output_nodes))

        ### Set self.activation_function to the choice of function
        def relu(x):
            return x * (x > 0)

        self.activation_function = relu


    def train(self, X, dia, row_id, epsi, MTR_DuctArea, target_df, zeta_col_names,V_max_org):
        ''' Train the network on batch of input features. In this version, the
        function outputs two correction factors.

            Arguments
            ---------
            X:             2D array of input features, each row is one data record, each column is a feature
            dia:           diameter input for calculating Zeta using the `SHR_Zeta_3D` function
            row_id:        row index of the training data in the current batch
            epsi:          Modified Correction factor for MTR
            MTR_DuctArea:  Duct Area for the Mixer Tapping Restrictor
            target_df:     Source dataframe containing required columns
            zeta_col_names:Zeta names of all the elements
            V_max_org:     Max Value of Flow Rate column
        '''
        n_records = 1 # Because we pass only one data point in every batch

        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape) # Create empty array filled with zeros
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape) # Create empty array filled with zeros

        ### FORWARD pass ###
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        c_f = final_inputs # Correction_factor from final output layer having linear (identity) activation function

        if (c_f[0] < 0):
            cf_epsi = c_f - epsi
            c_f = abs(c_f)
            print('\nNN output is -ve. Taking |(c_f)| value:',c_f)
        elif (c_f[0] > 0):
            print('\nNN output (C_f):',c_f[0])
            cf_epsi = c_f + epsi


         ## Pass Neural network output (correction factor) to compute new Zeta and then re-run FDDN solver
        _,target_df[MTR+'_Zeta3D'].loc[row_id] = zip(*[SHR_Zeta_3D(1,dia,MTR_DuctArea,c_f)])
        print('New_Zeta:',target_df[MTR+'_Zeta3D'].loc[row_id].values)

        ## Call FDDN Solver to compute new flowrates with the updated Zeta
        mixp_input,ambp_input,ambt_input,zeta_input = fddn_zeta_input(target_df[['HoV','MIXP','AMBP','AMBT','TZ6_Flow']+zeta_col_names].loc[row_id])
        FDDN_FlowRates_raw_df = FDDN_Solver(mixp_input,ambp_input,ambt_input,zeta_input)
        FDDN_FlowRates_df = FDDN_output_df_gen(FDDN_FlowRates_raw_df )
        V_hat_FDDN = FDDN_FlowRates_df['TZ6_Flow'].values
        V_hat_FDDN_scaled = V_hat_FDDN/V_max_org

        ## Pass Neural network output (correction factor) to compute new Zeta and then re-run FDDN solver
        _,target_df[MTR+'_Zeta3D'].loc[row_id] = zip(*[SHR_Zeta_3D(1,dia,MTR_DuctArea,cf_epsi)])
        print('New_Zeta_epsi:',target_df[MTR+'_Zeta3D'].loc[row_id].values)

         ## Call FDDN Solver to compute new flowrates with the updated Zeta with Epsilon
        mixp_input_epsi,ambp_input_epsi,ambt_input_epsi,zeta_input_epsi = fddn_zeta_input(target_df[['HoV','MIXP','AMBP','AMBT','TZ6_Flow']+zeta_col_names].loc[row_id])
        FDDN_FlowRates_raw_df_epsi = FDDN_Solver(mixp_input_epsi,ambp_input_epsi,ambt_input_epsi,zeta_input_epsi)
        FDDN_FlowRates_df_epsi = FDDN_output_df_gen(FDDN_FlowRates_raw_df_epsi )
        V_hat_epsi_FDDN = FDDN_FlowRates_df_epsi['TZ6_Flow'].values
        V_hat_epsi_FDDN_scaled = V_hat_epsi_FDDN/V_max_org

        HoV = target_df['HoV'].loc[row_id].values
        V_true_LTR = target_df['TZ6_Flow'].loc[row_id].values

        print('Row ID:',row_id)
        print('HoV:',HoV)
        print('FlowRate Difference (LTR - FDDN):',V_true_LTR - V_hat_FDDN,'l/s')


        ### BACKWARD pass ###
        # Custom Error Calculation
        V_max = 1.0 # as the data has already been normalized by max
        # Output error
        ## X[:,-1] --> Returns the last column from the feature dataset
        flowrate_diff = X[:,-1] - V_hat_FDDN_scaled # Difference between the FDDN flowrate and LTR Flow Rate
        error = (1/(2 * (V_max)**2)) * (flowrate_diff)**2 + (1/2) * (c_f - 1)**2

        # Derivative of flowrate with respect to neural network o/p (i.e correction factor)
        dv_da = (V_hat_epsi_FDDN - V_hat_FDDN) / epsi

        # Backpropagated error terms
        output_error_term = (1/(V_max)**2) * (flowrate_diff)*dv_da + (c_f - 1)

        # Hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T) ## This results in
        hidden_error_term = hidden_error * 1. * (hidden_outputs > 0) # For ReLu activation return it's derivative 1.*(h > 0)

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X.T

        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs.T

        # Update the weights
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden  += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

        ## Calculate Percentage Difference between flowrates for early stopping
        percentage_diff = 100 * 2 * (abs(V_true_LTR - V_hat_FDDN)) / (V_true_LTR + V_hat_FDDN)
        if (percentage_diff < 0.35): # % diff < 0.35%
            break_flag = 1
            print('% Diff between flowrates:',percentage_diff)
        else:
            break_flag = 0

        return HoV,V_hat_FDDN,V_true_LTR,break_flag,c_f[0]

    def predict(self, features,dia,row_id):
        ''' Run a forward pass through the network with input features to make predictions.

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # Hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        c_f = final_inputs # signals from final output layer
        if (c_f[0] < 0):
            c_f = abs(c_f)
            print('\nNN output is -ve. Taking ABS(c_f) value:',c_f)
        elif (c_f[0] > 0):
            print('\nNN output (C_f):',c_f[0])
        return c_f
