###
# Author: Dilip Rajkumar
# This py script contains all the functions required for calling the FDDN Solver in batch Mode
# and storing all the results in a dataframe
###

import pandas as pd
import numpy as np
import json
from subprocess import Popen

## Declare global variables
cols_to_drop = ['TZ-6_zmix', 'MTR-600', 'riser_LAO_RHS', 'riser_LAO_LHS', 'riser_CAO_RHS', 'riser_CAO_LHS', 'R-610', 'R-611', 'R-612', 'R-613',
                'CAO-620_fwd', 'CAO-620_mid', 'CAO-620_aft', 'CAO-621_fwd', 'CAO-621_mid', 'CAO-621_aft',
                'LAO-630', 'LAO-632', 'LAO-634', 'LAO-636', 'LAO-638', 'LAO-640', 'LAO-631', 'LAO-633', 'LAO-635', 'LAO-637', 'LAO-639', 'LAO-641' ,
                'pp2610pdloss', 'pp2611pdloss', 'pp2612pdloss', 'pp2613pdloss', '26120-R620_F', '26120-R620_M', '26120-R620_A', '26121-R621_F', '26121-R621_M', '26121-R621_A']

CAO_LH_cols = ['CAOLH_C66-C68','CAOLH_C68-C70','CAOLH_C70-C72','CAOLH_C72-C74','CAOLH_C74-C76','CAOLH_C76-C78']
CAO_RH_cols = ['CAORH_C66-C68','CAORH_C68-C70','CAORH_C70-C72','CAORH_C72-C74','CAORH_C74-C76','CAORH_C76-C78']
LAO_LH_cols = ['LAOLH_C66-C68','LAOLH_C68-C70','LAOLH_C70-C72','LAOLH_C72-C74','LAOLH_C74-C76','LAOLH_C76-C78']
LAO_RH_cols = ['LAORH_C66-C68','LAORH_C68-C70','LAORH_C70-C72','LAORH_C72-C74','LAORH_C74-C76','LAORH_C76-C78']

renamed_cols = {'CAOR-621_fwd':'CAOLH_C66-C68','CAOR-621_mid':'CAOLH_C70-C72','CAOR-621_aft':'CAOLH_C72-C74','CAOR-620_fwd':'CAORH_C66-C68','CAOR-620_mid':'CAORH_C70-C72','CAOR-620_aft':'CAORH_C72-C74',
'LAOR-631':'LAOLH_C66-C68','LAOR-633':'LAOLH_C68-C70','LAOR-635':'LAOLH_C70-C72','LAOR-637':'LAOLH_C72-C74','LAOR-639':'LAOLH_C74-C76','LAOR-641':'LAOLH_C76-C78',
'LAOR-630':'LAORH_C66-C68','LAOR-632':'LAORH_C68-C70','LAOR-634':'LAORH_C70-C72','LAOR-636':'LAORH_C72-C74','LAOR-638':'LAORH_C74-C76','LAOR-640':'LAORH_C76-C78' }

cols_order = ['CAOLH_C66-C68','CAOLH_C70-C72','CAOLH_C72-C74','CAORH_C66-C68','CAORH_C70-C72','CAORH_C72-C74'] + LAO_LH_cols + LAO_RH_cols

# Copy Flowrate from one frame to another
source_CAOR_columns = ['CAOLH_C66-C68','CAOLH_C72-C74','CAOLH_C74-C76','CAORH_C66-C68','CAORH_C72-C74','CAORH_C74-C76']
target_CAOR_columns = ['CAOLH_C68-C70','CAOLH_C74-C76','CAOLH_C76-C78','CAORH_C68-C70','CAORH_C74-C76','CAORH_C76-C78']

flowrate_by3_cols = ['CAOLH_C72-C74','CAOLH_C74-C76','CAOLH_C76-C78','CAORH_C72-C74','CAORH_C74-C76','CAORH_C76-C78']
flowrate_by2_cols = ['CAOLH_C66-C68','CAOLH_C68-C70','CAORH_C66-C68','CAORH_C68-C70']


# Read the FDDN JSON file through default json library
with open('MSN_TZ6.fddn','r') as f: 
    json_data = json.load(f)

name_variables = []
zeta_values = []

for i in range(len(json_data['Model1']['Elements'])):
    element_name = json_data['Model1']['Elements'][i]['Values'][1]['Value']
    zeta = json_data['Model1']['Elements'][i]['Values'][12]['Value']
    name_variables.append(element_name)
    zeta_values.append(zeta)

def fddn_zeta_input(df):
    mixp_input = df.iloc[:,1].values.tolist()
    ambp_input = df.iloc[:,2].values.tolist()
    ambt_input = df.iloc[:,3].values.tolist()
    zeta_input = df.iloc[:,5:].values.tolist()
    return mixp_input,ambp_input,ambt_input,zeta_input

def FDDN_Solver(select_elements,mixp_list,ambp_list,ambt_list,zeta_list):
    '''
    Modifies the FDDN JSON file with new values.
    Executes the FDDN solver in batch mode giving the new datapoints with zeta values as input and predicts flow rates.
    Reads the saved results from the FDDN Solver and appends into a raw dataframe.
    '''
    FDDN_V_df = pd.DataFrame() #Initialize an Empty dataframe
#     print('Output Column Names:\n',cols_order)
    indices = [i for i in range(len(name_variables)) if name_variables[i] in select_elements]
    for i in range(len(zeta_list)):
        new_zeta = list( map(str, zeta_list[i]) )
        for (index, value) in zip(indices, new_zeta):
            zeta_values[index] = value #Replace Zeta at specific locations in the zeta_values list with new values from every ASD point
        # EDIT JSON file - Modify Ambient conditions
        json_data['Config']['Values'][11]['Value'] = str(ambp_list[i])
        json_data['Model1']['Elements'][0]['Values'][15]['Value'] = str(mixp_list[i])
        json_data['Model1']['Elements'][0]['Values'][19]['Value'] = str(ambt_list[i])
        for j in range(len(zeta_values)): # EDIT JSON file - Modify all Zeta Values
            json_data['Model1']['Elements'][j]['Values'][12]['Value'] = zeta_values[j]
        with open('TZ6_LTR_Vhat.fddn', 'w') as f: # Save out FDDN json file containing input values to the FDDN Solver
            f.write(json.dumps(json_data,indent=2))
        Popen("TZ6_run_fddn.bat", stdout=None, stderr = None).communicate() # Run FDDN Solver
        # Read Saved results from the FDDN Solver - Update filename in 'run_fddn.bat'
        fddn_results_df = pd.read_csv('TZ6_LTR_Vhat.result_HYBRD1',delimiter=';',header=[0,1])
        fddn_results_df.columns = fddn_results_df.columns.map(''.join)
        fddn_results_df.rename(index=str, columns={"NameString": "#", "masskg/s": "MassFlow(kg/s)","rho1kg/m3":"rho1(kg/m³)","rho2kg/m3":"rho2(kg/m³)","V1m3/s":"V1(m³/s)"},inplace = True)
        fddn_results_df['Density(kg/m³)'] = (fddn_results_df['PStat1Pa']+ ambp_list[i])/(287*ambt_list[i])
        fddn_results_df['Flow_Rate(m³/s)'] = (fddn_results_df['MassFlow(kg/s)'])/(fddn_results_df['Density(kg/m³)'])#Convert MassFlow to FlowRate
        df_temp = fddn_results_df[['#','Flow_Rate(m³/s)']].set_index('#').T
        df_temp.drop(columns = cols_to_drop, inplace = True)
        df_temp.rename(columns=renamed_cols, inplace = True)
        # print('FDDN Solver Output:',np.around(np.array(1000*df_temp[cols_order].values),3))
        FDDN_V_df = FDDN_V_df.append(df_temp[cols_order] )
        FDDN_V_df.reset_index(drop=True,inplace = True)
    return FDDN_V_df.mul(1000) #Convert values in cu.m/s to l/s


def FDDN_output_df_gen(raw_fddn_df):
    '''
    Transforms the raw fddn output dataframe into a modified dataframe as required.
    '''
    for source_column,target_column in zip(source_CAOR_columns,target_CAOR_columns):
        idx = raw_fddn_df.columns.get_loc(source_column)
        raw_fddn_df.insert(loc=(idx), column=target_column, value= raw_fddn_df[source_column])
        idx+=1
    ## Cols where FlowRate is / 3
    raw_fddn_df[flowrate_by3_cols] = raw_fddn_df[flowrate_by3_cols] / 3
    ## Cols where FlowRate is / 2
    raw_fddn_df[flowrate_by2_cols] = raw_fddn_df[flowrate_by2_cols] / 2
    # Compute Summed Flows
    raw_fddn_df['CAOLH_SumFlow'] = raw_fddn_df[CAO_LH_cols].sum(axis=1) # Compute total flow rate by summing all CAO_LH flow rates
    raw_fddn_df['CAORH_SumFlow'] = raw_fddn_df[CAO_RH_cols].sum(axis=1) # Compute total flow rate by summing all CAO_RH flow rates
    raw_fddn_df['LAOLH_SumFlow'] = raw_fddn_df[LAO_LH_cols].sum(axis=1) # Compute total flow rate by summing all LAO_LH flow rates
    raw_fddn_df['LAORH_SumFlow'] = raw_fddn_df[LAO_RH_cols].sum(axis=1) # Compute total flow rate by summing all LAO_RH flow rates
    raw_fddn_df['TZ6_Flow'] = raw_fddn_df[CAO_LH_cols+CAO_RH_cols+LAO_LH_cols+LAO_RH_cols].sum(axis=1)
    return raw_fddn_df
