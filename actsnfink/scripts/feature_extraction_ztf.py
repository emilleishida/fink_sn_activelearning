# Copyright 2025 
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

import pandas as pd
import numpy as np
from copy import deepcopy

from actsnfink.classifier_sigmoid import get_sigmoid_features_dev
from fink_utils.data.utils import format_data_as_snana
from actsnfink.classifier_sigmoid import RF_FEATURE_NAMES

__all__ = [
    'concat_val',
    'apply_selection_cuts_ztf',
    'extract_features_rf_snia'
]


def concat_val(df, colname: str):
    """Concatenate historical and current measurements for 1 alert.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing data for 1 alert
    colname: str
        Name of the column to concatenate.
    prefix: str
        Additional prefix to add to the column name. Default is 'c'.

    Returns
    -------
    hist_vals: list
        list  containing the concatenation of historical and current measurements.        
    """
    
    current_val = [df["candidate"].get(colname)]
    
    prv = df.get("prv_candidates", None)
    
    if prv is not None:
        hist_vals = [p.get(colname) for p in prv]
    else:
        hist_vals = []
        
    return hist_vals

def apply_selection_cuts_ztf(
    magpsf: pd.Series,
    ndethist: pd.Series,
    #cdsxmatch: pd.Series,
    minpoints: int = 4,
    maxndethist: int = 20,
) -> pd.Series:
    """Apply selection cuts to keep only alerts of interest for early SN Ia analysis

    Parameters
    ----------
    magpsf: pd.Series
        Series containing data measurement (array of double). Each row contains
        all measurement values for one alert.
    ndethist: pd.Series
        Series containing length of the alert history (int).
        Each row contains the (single) length of the alert.
    #cdsxmatch: pd.Series
    #    Series containing crossmatch label with SIMBAD (str).
    #    Each row contains one label.

    Returns
    -------
    mask: pd.Series
        Series containing `True` if the alert is valid, `False` otherwise.
        Each row contains one boolean.
    """
    # Flag alerts with less than 3 points in total
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) >= minpoints

    # only alerts with less or equal than 20 measurements
    mask *= ndethist.astype(int) <= maxndethist

    # reject galactic objects
    #list_of_sn_host = return_list_of_eg_host()
    #mask *= cdsxmatch.apply(lambda x: x in list_of_sn_host)

    return mask

def extract_features_rf_snia(
    jd,
    fid,
    magpsf,
    sigmapsf,
    #cdsxmatch,
    ndethist,
    min_rising_points=None,
    min_data_points=None,
    rising_criteria=None,
) -> pd.Series:
    """Return the features used by the RF classifier.

    There are 12 features. Order is:
    a_g,b_g,c_g,snratio_g,chisq_g,nrise_g,
    a_r,b_r,c_r,snratio_r,chisq_r,nrise_r

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    #cdsxmatch: Spark DataFrame Column
    #    Type of object found in Simbad (string)
    ndethist: Spark DataFrame Column
        Column containing the number of detection by ZTF at 3 sigma (int)
    min_rising_points, min_data_points: int
        Parameters from fink_sn_activelearning.git
    rising_criteria: str
        How to compute derivatives: ewma (default), or diff.

    Returns
    -------
    features: list of str
        List of string.

    Examples
    --------
    >>> df = pd.read_parquet(ztf_alert_sample)

    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    for colname in what:
        df[prefix + colname] = df.apply(concat_val, args=[colname], axis=1)

    # Expose extra parameter
    ndethist = pd.Series([df['candidate'][i]['ndethist'] for i in range(df.shape[0])])

    # Perform the fit + classification (default model)
    >>> features = df.apply(extract_features_rf_snia, axis=1, 
    ...                           args=[df['cjd'], df['cfid'], df['cmagpsf'], df['csigmapsf', 'ndethist']])

    >>> for name in RF_FEATURE_NAMES:
    ...   index = RF_FEATURE_NAMES.index(name)
    ...   df[name] = features[:,index]

    # Trigger something
    >>> sum(df[RF_FEATURES_NAMES[0]] != 0) == 5
    True
    """
    if min_rising_points is None:
        min_rising_points = pd.Series([2])
    if min_data_points is None:
        min_data_points = pd.Series([4])
    if rising_criteria is None:
        rising_criteria = pd.Series(["ewma"])

    mask = apply_selection_cuts_ztf(magpsf, ndethist) #, cdsxmatch)

    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    candid = pd.Series(range(len(jd)))
    pdf = format_data_as_snana(jd, magpsf, sigmapsf, fid, candid, mask)

    test_features = []
    for id in np.unique(pdf["SNID"]):
        pdf_sub = pdf[pdf["SNID"] == id]
        features = get_sigmoid_features_dev(
            pdf_sub,
            min_rising_points=min_rising_points.to_numpy()[0],
            min_data_points=min_data_points.to_numpy()[0],
            rising_criteria=rising_criteria.to_numpy()[0],
        )
        test_features.append(features)

    to_return_features = np.zeros((len(jd), len(RF_FEATURE_NAMES)), dtype=float)
    to_return_features[mask] = test_features

    return np.array(to_return_features)

def main():

    pre_data_test = '/media/emille/git/Fink/supernova_al/repository/' + \
                       'fink_sn_activelearning/data/test_alerts.parquet'
    
    # read data
    data = pd.read_parquet(pre_data_test)
    
    # Required alert columns
    what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    prefix = 'c'
    what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    for colname in what:
        data[prefix + colname] = data.apply(concat_val, args=[colname], axis=1)

    # expose feature from outside candidates
    ndethist = pd.Series([data['candidate'][i]['ndethist'] for i in range(data.shape[0])])

    # extract features
    features = extract_features_rf_snia(data['cjd'], data['cfid'], 
                                        data['cmagpsf'], data['csigmapsf'], ndethist)

    print('Found ', sum(features[:,0] != 0), 'valid features.')
    print('Correct answer is 5.')

    
    
    return None
    
if __name__ == '__main__':
    main()
    