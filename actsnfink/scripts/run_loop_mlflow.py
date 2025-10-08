# Copyright 2022
# Author: Emille Ishida
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/mit-license.php
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from actsnfink import *

import os
import pandas as pd

def main():

    ################################################################
    
    #########     User choices: general    #########################
    
    create_matrix = False            # create raw data file by combining all TNS + a few simbad files
    n_files_simbad = 5              # number of simbad files randomly chosen to compose the raw data
    
    
    fname_features_matrix = '../../data/features.csv'             # output features file
    fname_raw_output = '../../../test_mlflow_2/data/raw.csv.gz'          # output raw data file
    dirname_input = '../../../data/AL_data/'                       # input directory with labelled alerts
    dirname_output = '../../../test_mlflow_2/'                       # root products output directory
    append_name = ''                                               # append to all metric, prob and queries names
    
    nloops = 3                         # number of learning loops
    strategy = 'UncSampling'            # query strategy
    initial_training = 10               # total number of objs in initial training
    frac_Ia_tot = 0.5                   # fraction of Ia in initial training 
    n_realizations = 1                  # total number of realizations
    n_realizations_ini = 0              # start from this realization number
    new_raw_file = False                 # save raw data in one file
    input_raw_file = fname_raw_output   # name of raw data file
    n = 15000                           # number of random simbad objects per file 
                                        # to be used as part of the raw data

    mlflow_uri = "https://mlflow-dev.fink-broker.org"     # address of mlflow server
    mlflow_exp = 'finksnclass_ztf_evaluate'          # root name for this experiment run
    
    drop_zeros = True                   # ignore objects with observations in only 1 filter
    screen = True                       # print debug comments to screen
    
    #####  User choices: For Figure 7      ##########################
    
    initial_state_from_file = False      # read initial state from a fixed file
    initial_state_version = 68            # version from which initial state is chosen
    
    ################################################################
    ################################################################
    
    features_names = ['a_g', 'b_g', 'c_g', 'snratio_g', 'mse_g', 'nrise_g', 
                          'a_r', 'b_r', 'c_r', 'snratio_r', 'mse_r', 'nrise_r']
    
    for name in [dirname_output + '/', 
                 dirname_output + '/data/', 
                 dirname_output + '/' + strategy + '/', 
                 dirname_output + '/' + strategy + '/class_prob/',
                 dirname_output + '/' + strategy + '/metrics/', 
                 dirname_output + '/' + strategy + '/queries/',
                 dirname_output + '/' + strategy + '/training_samples/', 
                 dirname_output + '/' + strategy + '/test_samples/']:
        if not os.path.isdir(name):
            os.makedirs(name)  
    
    if create_matrix:
        matrix_clean = build_matrix(fname_output=fname_features_matrix, dirname_input=dirname_input, dirname_output=dirname_output + 'data/',
                                    fname_raw_output=fname_raw_output, new_raw_file=new_raw_file,
                                    input_raw_file=input_raw_file,n=n,
                                   n_files_simbad=n_files_simbad, drop_zeros=drop_zeros, screen=screen)
        print(np.unique(matrix_clean['type'].values))
        
    else:
        matrix_clean = pd.read_csv(fname_features_matrix, comment='#')    
    
    if initial_state_from_file:
        fname_ini_train = dirname_output + '/UncSampling/training_samples/initialtrain_v' + str(initial_state_version) + '.csv'              
        fname_ini_test = dirname_output + '/UncSampling/test_samples/initial_test_v' + str(initial_state_version) + '.csv'
    
        output_metrics_file = dirname_output + '/' + strategy + '/metrics/metrics_' + strategy + '_v' + str(initial_state_version) + append_name + '.dat'
        output_queried_file = dirname_output + '/' + strategy + '/queries/queried_' + strategy + '_v'+ str(initial_state_version) + append_name + '.dat'
        output_prob_root = dirname_output + '/' + strategy + '/class_prob/v' + str(initial_state_version) + '/class_prob_' + strategy + append_name
    
        name = dirname_output + '/' + strategy + '/class_prob/v' + str(initial_state_version) + '/'
        if not os.path.isdir(name):
            os.makedirs(name)
        data = read_initial_samples(fname_ini_train, fname_ini_test)
        
        # perform learnin loop
        learn_loop(data, nloops=nloops, strategy=strategy, 
                   output_metrics_file=output_metrics_file, 
                   output_queried_file=output_queried_file,
                   classifier='RandomForest', seed=None,
                   batch=1, screen=True, output_prob_root=output_prob_root,
                   mlflow_uri=mlflow_uri, mlflow_exp=mlflow_exp)
        
    else:
        for v in range(n_realizations_ini, n_realizations):
            output_metrics_file = dirname_output + '/' + strategy + '/metrics/metrics_' + strategy + '_v' + str(v) + append_name + '.dat'
            output_queried_file = dirname_output + '/' + strategy + '/queries/queried_' + strategy + '_v'+ str(v) + append_name + '.dat'
            output_prob_root = dirname_output + '/' + strategy + '/class_prob/v' + str(v) + '/class_prob_' + strategy + append_name
    
            name = dirname_output + '/' + strategy + '/class_prob/v' + str(v) + '/'
            if not os.path.isdir(name):
                os.makedirs(name)
            #build samples        
            data = build_samples(matrix_clean, initial_training=initial_training, screen=True)
        
            # save initial data        
            train = pd.DataFrame(data.train_features, columns=features_names)
            train['objectId'] = data.train_metadata['id'].values
            train['type'] = data.train_metadata['type'].values
            train.to_csv(dirname_output + '/' + strategy + '/training_samples/initialtrain_v' + str(v) + '.csv', index=False)
        
            test = pd.DataFrame(data.test_features, columns=features_names)
            test['objectId'] = data.test_metadata['id'].values
            test['type'] = data.test_metadata['type'].values
            test.to_csv(dirname_output + '/' + strategy + '/test_samples/initial_test_v' + str(v) + '.csv', index=False)        
    
            # perform learnin loop
            learn_loop(data, nloops=nloops, strategy=strategy, 
                   output_metrics_file=output_metrics_file, 
                   output_queried_file=output_queried_file,
                   classifier='RandomForest', seed=None,
                   batch=1, screen=True, output_prob_root=output_prob_root, mlflow_uri=mlflow_uri, 
                   mlflow_exp=mlflow_exp, features_names=features_names)
    
if __name__ == '__main__':
    main()

