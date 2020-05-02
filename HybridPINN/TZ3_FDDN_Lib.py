###
# Author: Dilip Rajkumar
# This python script contains all the functions required for calling the FDDN Solver in batch Mode
# and storing all the results in a dataframe
###

import pandas as pd
import numpy as np
import json
from subprocess import Popen

## Declare global variables
cols_to_drop = ['TZ-3_zmix', 'MTR-300', 'riser_LAO_RHS', 'riser_LAO_LHS', 'riser_CAO_RHS', 'riser_CAO_LHS', 'R-310', 'R-311', 'R-312', 'R-313',
               'CAO-320_fwd', 'CAO-320_mid', 'CAO-320_aft', 'CAO-321_fwd', 'CAO-321_mid', 'CAO-321_aft', 'LAO-330', 'LAO-332', 'LAO-334', 'LAO-336', 'LAO-331', 'LAO-333', 'LAO-335', 'LAO-337',
               'pp2310pdloss', 'pp2311pdloss', 'pp2312pdloss', 'pp2313pdloss', '23120-320_F', '23120-320_M', '23120-320_A', '23130-321_F', '23130-321_M', '23130-321_A',
                '3302.2-3303', '3303-3304', '3302.2-3305', '3305-3306', '2301-3302.1', '2301-3302.2','CAOR-320_aft','CAOR-321_aft']


CAO_LH_cols = ['CAOLH_C38-C40','CAOLH_C40-C42','CAOLH_C42-C44','CAOLH_C44-C46']
CAO_RH_cols = ['CAORH_C38-C40','CAORH_C40-C42','CAORH_C42-C44','CAORH_C44-C46']
LAO_LH_cols = ['LAOLH_C38-C40','LAOLH_C40-C42','LAOLH_C42-C44','LAOLH_C44-C46']
LAO_RH_cols = ['LAORH_C38-C40','LAORH_C40-C42','LAORH_C42-C44','LAORH_C44-C46']

renamed_cols = {'CAOR-321_fwd':'CAOLH_C38-C40','CAOR-321_mid':'CAOLH_C44-C46',
                'CAOR-320_fwd':'CAORH_C38-C40','CAOR-320_mid':'CAORH_C44-C46',
                'LAOR-331':'LAOLH_C38-C40','LAOR-333':'LAOLH_C40-C42','LAOR-335':'LAOLH_C42-C44','LAOR-337':'LAOLH_C44-C46',
                'LAOR-330':'LAORH_C38-C40','LAOR-332':'LAORH_C40-C42','LAOR-334':'LAORH_C42-C44','LAOR-336':'LAORH_C44-C46'}

cols_order = ['CAOLH_C38-C40','CAOLH_C44-C46','CAORH_C38-C40','CAORH_C44-C46'] + LAO_LH_cols + LAO_RH_cols

# Read the FDDN JSON file through default json library
with open('MSN_TZ3.fddn','r') as f:
    json_data = json.load(f)

zeta_values = []
name_variables  = []
fm_flag_values  = []
fp2_flag_values = []

for i in range(len(json_data['Model1']['Elements'])):
    element_name = json_data['Model1']['Elements'][i]['Values'][1]['Value']
    fm_flag = json_data['Model1']['Elements'][i]['Values'][4]['Value']
    zeta = json_data['Model1']['Elements'][i]['Values'][12]['Value']
    fp2_flag = json_data['Model1']['Elements'][i]['Values'][16]['Value']
    name_variables.append(element_name)
    zeta_values.append(zeta)
    fm_flag_values.append(fm_flag)
    fp2_flag_values.append(fp2_flag)

def fddn_zeta_flag_input(df):
    mixp_input = df.iloc[:,1].values.tolist()
    ambp_input = df.iloc[:,2].values.tolist()
    ambt_input = df.iloc[:,3].values.tolist()
    fm_flag_input = df.iloc[:,4:10].values.tolist()
    fp2_flag_input = df.iloc[:,10:16].values.tolist()
    zeta_input = df.iloc[:,16:].values.tolist()
    return mixp_input, ambp_input, ambt_input, zeta_input, fm_flag_input, fp2_flag_input

def FDDN_Solver(zeta_select_elements, fm_flag_select_elements, mixp_list, ambp_list, ambt_list, zeta_list, fm_flag_list, fp2_flag_list ):
    '''
    Modifies the FDDN JSON file with new values.
    Executes the FDDN solver in batch mode giving the new datapoints with zeta values as input and predicts flow rates.
    Reads the saved results from the FDDN Solver and appends into a raw dataframe.
    '''
    FDDN_V_df = pd.DataFrame() #Initialize an Empty dataframe
#     print('Output Column Names:\n',cols_order)
    indices = [i for i in range(len(name_variables)) if name_variables[i] in zeta_select_elements]
    fm_flag_indices = [i for i in range(len(name_variables)) if name_variables[i] in fm_flag_select_elements]
    for i in range(len(zeta_list)):
        new_zeta     = list(map(str, zeta_list[i]))
        new_fm_flag  = list(map(str, fm_flag_list[i]))
        new_fp2_flag = list(map(str, fp2_flag_list[i]))
        # Replace Zeta, FM and FP2 flags at specific locations in the zeta_values, fm_flag_values, fp2_flag_values list with new values from every data point
        for (index, value) in zip(indices, new_zeta):
            zeta_values[index] = value
        for (ind, fm_flag, fp2_flag) in zip(fm_flag_indices, new_fm_flag, new_fp2_flag):
            fm_flag_values[ind] = fm_flag
            fp2_flag_values[ind] = fp2_flag

        # EDIT JSON file - Modify Ambient conditions
        json_data['Config']['Values'][11]['Value'] = str(ambp_list[i])
        json_data['Model1']['Elements'][0]['Values'][15]['Value'] = str(mixp_list[i])
        json_data['Model1']['Elements'][0]['Values'][19]['Value'] = str(ambt_list[i])
        for j in range(len(zeta_values)): # EDIT JSON file - Modify all Zeta Values
            json_data['Model1']['Elements'][j]['Values'][12]['Value'] = zeta_values[j]
        ## Print original Zeta Values in json sheet for verification
#         print("\nZETA Values written in .FDDN file")
#         for e1 in json_data['Model1']['Elements']:
#             for element in e1['Values']:
#                 if element['Name'] == "Name":
#                     fddn_ele_name = element['Value']
#                 if element['Name'] == "Zeta":
#                     loss_coefficient_value = element['Value']
#                     print(fddn_ele_name, ':' ,loss_coefficient_value)
        for k in range(len(fm_flag_values)):
            json_data['Model1']['Elements'][k]['Values'][4]['Value'] = fm_flag_values[k]
            json_data['Model1']['Elements'][k]['Values'][16]['Value'] = fp2_flag_values[k]
        # Print original FM Flag Values for verification
#         print("\nFM Flag Values written in .FDDN file")
#         for e1 in json_data['Model1']['Elements']:
#             for element in e1['Values']:
#                 if element['Name'] == "Name":
#                     fddn_ele_name = element['Value']
#                 if element['Name'] == "FM":
#                     fm_flag_value = element['Value']
#                     print(fddn_ele_name, ':' ,fm_flag_value)
        with open('TZ3_LTR_Vhat.fddn', 'w') as f: # Save out FDDN json file containing input values to the FDDN Solver
            f.write(json.dumps(json_data,indent=2))
        Popen("TZ3_run_fddn.bat", stdout=None, stderr = None).communicate() # Run FDDN Solver
        # Read Saved results from the FDDN Solver - Update filename in 'run_fddn.bat'
        fddn_results_df = pd.read_csv('TZ3_LTR_Vhat.result_HYBRD1',delimiter=';',header=[0,1])
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

# Copy Flowrate from one frame to another
source_CAOR_columns = ['CAOLH_C38-C40','CAOLH_C38-C40','CAORH_C38-C40','CAORH_C38-C40']
target_CAOR_columns = ['CAOLH_C40-C42','CAOLH_C42-C44','CAORH_C40-C42','CAORH_C42-C44']

flowrate_by3_cols = ['CAOLH_C38-C40','CAOLH_C40-C42','CAOLH_C42-C44', 'CAORH_C38-C40','CAORH_C40-C42','CAORH_C42-C44']

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

    # Compute Summed Flows
    raw_fddn_df['CAOLH_SumFlow'] = raw_fddn_df[CAO_LH_cols].sum(axis=1) # Compute total flow rate by summing all CAO_LH flow rates
    raw_fddn_df['CAORH_SumFlow'] = raw_fddn_df[CAO_RH_cols].sum(axis=1) # Compute total flow rate by summing all CAO_RH flow rates
    raw_fddn_df['LAOLH_SumFlow'] = raw_fddn_df[LAO_LH_cols].sum(axis=1) # Compute total flow rate by summing all LAO_LH flow rates
    raw_fddn_df['LAORH_SumFlow'] = raw_fddn_df[LAO_RH_cols].sum(axis=1) # Compute total flow rate by summing all LAO_RH flow rates
    raw_fddn_df['TZ3_Flow'] = raw_fddn_df[CAO_LH_cols+CAO_RH_cols+LAO_LH_cols+LAO_RH_cols].sum(axis=1)
    return raw_fddn_df
