# Copyright 2020-2021 
# Author: Marco Leoni and Emille Ishida
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

import actsnclass
import pandas as pd
import numpy as np
import os

from actsnclass import DataBase
from actsnfink.classifier_sigmoid import get_sigmoid_features_dev
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = [
    'build_matrix',
    'build_samples',
    'convert_full_dataset',
    'featurize_full_dataset',
    'learn_loop',
    'mag2fluxcal_snana',
    'read_initial_samples'
]


def mag2fluxcal_snana(magpsf: float, sigmapsf: float):
    """ Conversion from magnitude to Fluxcal from SNANA manual
    Parameters
    ----------
    magpsf: float
        PSF-fit magnitude from ZTF.
    sigmapsf: float
        Error on PSF-fit magnitude from ZTF. 
    
    Returns
    ----------
    fluxcal: float
        Flux cal as used by SNANA
    fluxcal_err: float
        Absolute error on fluxcal (the derivative has a minus sign)
    """
    if magpsf is None:
        return None, None
    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10 ** 10 * np.exp(-0.921034 * magpsf) * sigmapsf

    return fluxcal, fluxcal_err


def convert_full_dataset(pdf: pd.DataFrame, obj_id_header='candid',
                        keep_objid=False):
    """Convert an entire data set from mag to fluxcal.
    
    Parameters
    ----------
    pdf: pd.DataFrame
        Read directly from parquet files.
    obj_id_header: str (optional)
        Object identifier. Options are ['objectId', 'candid'].
        Default is 'candid'.
    keep_objid: bool (optional)
        If True, keep store 'objectId' beyond "obj_id_header". 
        Default is False
        
    Returns
    -------
    pd.DataFrame
        Columns are [obj_id_header, 'type', 'MJD', 'FLT', 
        'FLUXCAL', 'FLUXCALERR'].
    """
    # Ia types in TNS
    Ia_group = ['SN Ia', 'SN Ia-91T-like', 'SN Ia-91bg-like', 'SN Ia-CSM', 
                'SN Ia-pec', 'SN Iax[02cx-like]']
    
    # hard code ZTF filters
    filters = ['g', 'r']
    
    lc_flux_sig = []

    for index in range(pdf.shape[0]):

        objid = pdf['objectId'].values[index]
        name = pdf[obj_id_header].values[index]
        
        sntype_orig = pdf['TNS'].values[index]
        if sntype_orig == -99:
            sntype_orig = pdf['cdsxmatch'].values[index]
        
        if sntype_orig in Ia_group:
            sntype = 'Ia'
        else:
            sntype = str(sntype_orig).replace(" ", "")

        for f in range(1,3):
            
            if isinstance(pdf.iloc[index]['cfid'], str):
                ffs = np.array([int(item) for item in pdf.iloc[index]['cfid'][1:-1].split()])
                filter_flag = ffs == f
                mjd = np.array([float(item) for item in pdf.iloc[index]['cjd'][1:-1].split()])[filter_flag]
                mag = np.array([float(item) for item in pdf.iloc[index]['cmagpsf'][1:-1].split()])[filter_flag]
                magerr = np.array([float(item) for item in pdf.iloc[index]['csigmapsf'][1:-1].split()])[filter_flag]
            else:
                filter_flag = pdf['cfid'].values[index] == f
                mjd = pdf['cjd'].values[index][filter_flag]
                mag = pdf['cmagpsf'].values[index][filter_flag]
                magerr = pdf['csigmapsf'].values[index][filter_flag] 

            fluxcal = []
            fluxcal_err = []
            for k in range(len(mag)):
                f1, f1err = mag2fluxcal_snana(mag[k], magerr[k])
                fluxcal.append(f1)
                fluxcal_err.append(f1err)
        
            for i in range(len(fluxcal)):
                if keep_objid:
                    lc_flux_sig.append([objid, name, sntype, mjd[i], filters[f - 1],
                                    fluxcal[i], fluxcal_err[i]])
                else:
                    lc_flux_sig.append([name, sntype, mjd[i], filters[f - 1],
                                    fluxcal[i], fluxcal_err[i]])

    if keep_objid:
        names = ['objectId', 'id', 'type', 'MJD','FLT', 'FLUXCAL','FLUXCALERR']
    else:
        names = ['id', 'type', 'MJD','FLT', 'FLUXCAL','FLUXCALERR']
        
    lc_flux_sig = pd.DataFrame(lc_flux_sig, columns=names)

    return lc_flux_sig


def featurize_full_dataset(lc: pd.DataFrame, screen=False,
                           ewma_window=3, 
                           min_rising_points=2, 
                           min_data_points=4,
                           rising_criteria='ewma'):
    """Get complete feature matrix for all objects in the data set.
    
    Parameters
    ----------
    lc: pd.DataFrame
        Columns should be: ['objectId', 'type', 'MJD', 'FLT', 
        'FLUXCAL', 'FLUXCALERR'].
    screen: bool (optional)
        If True print on screen the index of light curve being fit.
        Default is False.
    ewma_window: int (optional)
        Width of the ewma window. Default is 3.
    min_rising_points: int (optional)
        Minimum number of rising points. Default is 1.
    min_data_points: int (optional)
        Minimum number of data points. Default is 3.
    rising_criteria: str (optional)
        Criteria for defining rising points. Options are 'diff' or 'ewma'.
        Default is 'ewma'.
        
    Returns
    -------
    pd.DataFrame
        Features for all objects in the data set. Columns are:
        ['objectId', 'type', 'a_g', 'b_g', 'c_g', 'snratio_g', 
        'mse_g', 'nrise_g', 'a_r', 'b_r', 'c_r', 'snratio_r',
        'mse_r', 'nrise_r']
    """
    
    # columns in output data matrix
    columns = ['id', 'type', 'a_g', 'b_g', 'c_g', 
               'snratio_g', 'mse_g', 'nrise_g', 'a_r', 'b_r', 'c_r',
               'snratio_r', 'mse_r', 'nrise_r']

    features_all = []

    for indx in range(np.unique(lc['id'].values).shape[0]):
 
        if screen:
            print('indx: ', indx)
        
        name = np.unique(lc['id'].values)[indx]

        obj_flag = lc['id'].values == name
        sntype = lc[obj_flag].iloc[0]['type']
    
        line = [name, sntype]
    
        features = get_sigmoid_features_dev(lc[obj_flag][['MJD',
                                                          'FLT',
                                                          'FLUXCAL',
                                                          'FLUXCALERR']],
                                           ewma_window=ewma_window, 
                                min_rising_points=min_rising_points, 
                                min_data_points=min_data_points,
                                rising_criteria=rising_criteria)
        
        for j in range(len(features)):
            line.append(features[j])
        
        features_all.append(line)
    
    feature_matrix = pd.DataFrame(features_all, columns=columns)

    return feature_matrix


# this was taken from https://github.com/COINtoolbox/ActSNClass/blob/master/actsnclass/database.py
def build_samples(features: pd.DataFrame, initial_training: int,
                 frac_Ia=0.5, screen=False):
    """Build initial samples for Active Learning loop.
    
    Parameters
    ----------
    features: pd.DataFrame
        Complete feature matrix. Columns are: ['objectId', 'type', 
        'a_g', 'b_g', 'c_g', 'snratio_g', 'mse_g', 'nrise_g', 
        'a_r', 'b_r', 'c_r', 'snratio_r', 'mse_r', 'nrise_r']
        
    initial_training: int
        Number of objects in the training sample.
    frac_Ia: float (optional)
        Fraction of Ia in training. Default is 0.5.
    screen: bool (optional)
        If True, print intermediary information to screen.
        Default is False.
        
    Returns
    -------
    actsnclass.DataBase
        DataBase for active learning loop
    """
    data = DataBase()
    
    # initialize the temporary label holder
    train_indexes = np.random.choice(np.arange(0, features.shape[0]),
                                     size=initial_training, replace=False)
    
    Ia_flag = features['type'].values == 'Ia'
    Ia_indx = np.arange(0, features.shape[0])[Ia_flag]
    nonIa_indx =  np.arange(0, features.shape[0])[~Ia_flag]
    
    indx_Ia_choice = np.random.choice(Ia_indx, size=max(1, initial_training // 2),
                                      replace=False)
    indx_nonIa_choice = np.random.choice(nonIa_indx, 
                        size=initial_training - max(1, initial_training // 2),
                        replace=False)
    train_indexes = list(indx_Ia_choice) + list(indx_nonIa_choice)
    
    temp_labels = features['type'].values[np.array(train_indexes)]

    if screen:
        print('\n temp_labels = ', temp_labels, '\n')

    # set training
    train_flag = np.array([item in train_indexes for item in range(features.shape[0])])
    
    train_Ia_flag = features['type'].values[train_flag] == 'Ia'
    data.train_labels = train_Ia_flag.astype(int)
    data.train_features = features[train_flag].values[:,2:]
    data.train_metadata = features[['id', 'type']][train_flag]
    
    # set test set as all objs apart from those in training
    test_indexes = np.array([i for i in range(features.shape[0])
                             if i not in train_indexes])
    test_ia_flag = features['type'].values[test_indexes] == 'Ia'
    data.test_labels = test_ia_flag.astype(int)
    data.test_features = features[~train_flag].values[:, 2:]
    data.test_metadata = features[['id', 'type']][~train_flag]
    
    # set metadata names
    data.metadata_names = ['id', 'type']
    
    # set everyone to queryable
    data.queryable_ids = data.test_metadata['id'].values
    
    if screen:
        print('Training set size: ', data.train_metadata.shape[0])
        print('Test set size: ', data.test_metadata.shape[0])
        print('  from which queryable: ', len(data.queryable_ids))
        
    return data


# This was slightly modified from https://github.com/COINtoolbox/ActSNClass/blob/master/actsnclass/learn_loop.py
def learn_loop(data: actsnclass.DataBase, nloops: int, strategy: str,
               output_metrics_file: str, output_queried_file: str,
               classifier='RandomForest', batch=1, screen=True, 
               output_prob_root=None, seed=42, nest=1000):
    """Perform the active learning loop. All results are saved to file.
    
    Parameters
    ----------
    data: actsnclass.DataBase
        Output from the build_samples function.
    nloops: int
        Number of active learning loops to run.
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    output_metrics_file: str
        Full path to output file to store metric values of each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently only 'RandomForest' is implemented.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    n_est: int (optional)
        Number of trees. Default is 1000.
    output_prob_root: str or None (optional)
        If str, root to file name where probabilities without extension!
        Default is None.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    seed: int (optional)
        Random seed.
    """

    for loop in range(nloops):

        if screen:
            print('Processing... ', loop)

        # classify
        data.classify(method=classifier, seed=seed, n_est=nest)
        
        if isinstance(output_prob_root, str):
            data_temp = data.test_metadata.copy(deep=True)
            data_temp['prob_Ia'] = data.classprob[:,1]
            data_temp.to_csv(output_prob_root + '_loop_' + str(loop) + '.csv', index=False)
            
        # calculate metrics
        data.evaluate_classification(screen=screen)

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch, seed=seed, screen=screen)
        print('indx: ', indx)
        
        # update training and test samples
        data.update_samples(indx, loop=loop)

        # save metrics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                          batch=batch, epoch=loop)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop, batch=batch,
                                 full_sample=False)
        
        
        
def build_matrix(fname_output: str, dirname_input: str, n: int,
                 dirname_output:str,
                fname_raw_output=None, new_raw_file=False, 
                input_raw_file=None, n_files_simbad=1,
                drop_zeros=False, screen=False):
    """Build full feature matrix to file.
    
    Parameters
    ----------
    fname_output: str
        Full path to output file.  
    dirname_input: str
        Full path to directory including all parquet files.
    n: int
        Number of simbad objects to choose.
    dirname_output: str
        Output directory to store the name of files used.
    drop_zeros: bool (optional)
        If True eliminate alerts with less than 3 points per filter. 
        Default is False. Filters with less than 3 points will have
        parameters equal to zero.
    fname_raw_output: str (optinal)
        Full path to filename containing raw data.
        Only used if new_raw_file == True.
    input_raw_file: str (optional)
        Full path to input raw data matrix.
        Only used if new_raw_file == False.
    new_raw_file: bool (optional)
        If True generate new input matrix from parquet files.
        Default is False.
    n_files_simbad: int (optional)
        Number of simbad files to use. Default is 1.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    
        
    Returns
    -------
    pd.DataFrame
        Features matrix.
    """
    used_files = []
    
    if new_raw_file:
        data_temp = []
        flist = os.listdir(dirname_input)
        simbad = 0
        tns = False
        # read all tns file and n_files_simbad random simbad file
        for name in flist:
            if ('simbad' in name and simbad < n_files_simbad):
                used_files.append(name)
                d1 = pd.read_parquet(dirname_input + name)
                data_temp.append(d1.sample(n, replace=False))
                simbad = simbad + 1
            elif 'tns' in name:
                used_files.append(name)
                d1 = pd.read_parquet(dirname_input + name)
                data_temp.append(d1)


        pdf7 = pd.concat(data_temp, ignore_index=True)
        pdf7.fillna(-99, inplace=True)
        pdf7.to_csv(fname_raw_output, index=False)
        
    else:
        pdf7 = pd.read_csv(input_raw_file, index_col=False)
        if ' ' in pdf7.keys()[0]:
            pdf7 = pd.read_csv(input_raw_file, delim_whitespace=True)
    
    # convert data to appropriate format
    lcs2 = convert_full_dataset(pdf7)

    # build feature matrix
    m = featurize_full_dataset(lcs2, screen=screen)

    # drop zeros
    if drop_zeros:
        m_final2 = m.replace(0, np.nan).dropna()
        m_final3 = m_final2.sample(frac=1).reset_index(drop=True)
        matrix = m_final3
    else:
        m_flag1 = m['a_g'].values == 0
        m_flag2 = m['b_g'].values == 0
        m_flag = np.logical_and(~m_flag1, ~m_flag2)
        matrix = m[m_flag]
    
    matrix.to_csv(fname_output, index=False)
    
    op1 = open(dirname_output + 'files_used_' + str(n_files_simbad) + 'f.txt', 'w')
    for item in used_files:
        op1.write(item + '\n')
    op1.close()

    print('Extract features from  ', matrix.shape[0], 'objects.')
    
    return matrix


def read_initial_samples(fname_train: str, fname_test:str):
    """Read initial training and test samples from file. 
    
    Parameters
    ----------
    fname_train: str
        Full path to training sample file.
    fname_test: str
        Full path to test sample file.
        
    Returns
    -------
    actsnclass.DataBase
        DataBase for active learning loop.    
    """
    
    # read data    
    data_train = pd.read_csv(fname_train)
    data_test = pd.read_csv(fname_test)

    # build DataBase object
    data = DataBase()
    data.metadata_names = ['id', 'type']
    
    data.train_labels = data_train.values[:,-1] == 'Ia'
    data.train_features = data_train.values[:,:-2]
    data.train_metadata = data_train[['objectId', 'type']]
    data.train_metadata.rename(columns={'objectId':'id'}, inplace=True)
    
    data.test_labels = data_test.values[:,-1] == 'Ia'
    data.test_features = data_test.values[:,:-2]
    data.test_metadata = data_test[['objectId', 'type']]
    data.test_metadata.rename(columns={'objectId':'id'}, inplace=True)
    
    data.queryable_ids = data.test_metadata[data.metadata_names[0]].values
    
    return data

def main():
    return None
    
if __name__ == '__main__':
    main()
