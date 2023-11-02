# Copyright 2020 AstroLab Software
# Author: Marco Leoni, Julien Peloton
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://opensource.org/licenses/mit-license.php
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import random
from copy import deepcopy

from actsnfink.sigmoid import fit_sigmoid
from actsnfink.sigmoid import delta_t
from actsnfink.sigmoid import compute_mse
from actsnfink.sigmoid import fsigmoid

__all__ = ['filter_data', 'mask_negative_data', 'get_fake_df', 'get_fake_fit_parameters',
          'get_fake_results', 'get_ewma_derivative', 'get_sn_ratio', 'get_predicted_flux',
          'get_data_to_export', 'get_diff', 'get_train_test', 'average_intraday_data', 'get_sigmoid_features_dev', 'get_sigmoid_features_elasticc',
          'get_sigmoid_features_elasticc_perfilter']


columns_to_keep = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
columns_to_keep_elastic = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']

fluxes = ['FLUXCAL', 'FLUXCALERR']
RF_FEATURE_NAMES = 'a_g,b_g,c_g,snratio_g,mse_g,nrise_g,a_r,b_r,c_r,snratio_r,mse_r,nrise_r'.split(',')


def filter_data(data, filt):
    """
    Select data according to the value of the
    filter (for ZTF only g, r)

    Parameters
    ----------
    data: pandas DataFrame
          containing light curves for different filters
    filt: string
          defining the filter (For ZTF, 'g' or 'r')

    Returns
    -------
    data_filt: pandas DataFrame
               containing light curves for the selected filter, filt
    """

    data_filt = deepcopy(data.loc[data['FLT'] == filt])

    return data_filt


def mask_negative_data(data, low_bound):
    """Mask data points whose FLUXCAL values are
       lower than a chosen lower bound

       Prameteres
       ----------
       data: pandas DataFrame
       light curve data for given filter
       low_bound: float
       minimum allowed value of flux

       Returns
       -------
       data: pandas DataFrame
       light curve with masked flux

        """
    
    masked_data = deepcopy(data[data['FLUXCAL'].values > low_bound])
    masked_data.dropna(inplace=True)
    
    return masked_data


def get_fake_df(filt):
    """Get fake data frame.

    Parameters
    ----------
    filt: char
    name of the desired filter

    Returns
    ------
    fake: pandas DataFrame with fake values

    """

    fake = pd.DataFrame({
        'VARLIST:': np.array(['OBS:', 'OBS:', 'OBS:']),
        'MJD': np.array([1, 2, 3]),
        'FLT': np.array([filt, filt, filt]),
        'FIELD': np.array([0.0, 0.0, 0.0]),
        'FLUXCAL': np.array([0, 0, 0]),
        'FLUXCALERR': np.array([1, 1, 1])
    })

    return fake


def get_fake_fit_parameters():
    """Function returning fake values
    for the fit parameters

    Returns
    -------
    [a, b, c, snratio, mse] = list of floats
    list of fake features in the absence of real fit

    """

    [a, b, c, snratio, mse] = [0, 0, 0, 0.1, 1e8]

    return [a, b, c, snratio, mse]


def get_fake_results(filt):
    """Returns fake results in case of problems with the fit

    Parameters
    ----------
    filt:char, the filter name (g, r)

    Returns
    ------
    [export, a, b, c, snratio, mse, nrise]: list of
        DataFrame and floats
    list of fake results in the absence of a real fit

    """
    nrise = 0
    [a, b, c, snratio, mse] = get_fake_fit_parameters()

    return [a, b, c, snratio, mse, nrise]


def get_ewma_derivative(data, ewma_window):
    """Compute the ewma and the derivative

    Parameters
    ----------
    data: pandas DataFrame
    ewma_window: width of the window
    in the rolling average

    Return
    ------
    ewma_derivative: pandas DataFrame
    derivative of the ewma of data

    """

    ewma_derivative = data.ewm(ewma_window, min_periods=0).mean().diff()

    return ewma_derivative

def get_diff(data):
    """Difference between each pair of points.

    Parameters
    ----------
    data: pandas DataFrame for FLUXCAL

    Return
    ------
    diff: pandas DataFrame
        difference between consecutive data points

    """

    diff = data.diff()

    return diff

def get_sn_ratio(data, data_err):
    """Compute signal to noise ratio

    Parameters
    ----------
    data: np.array
     rising flux, 'FLUXCAL'
    data_err: np.array
     error in the rising flyx, 'FLUXCALERR'

    Returns
    -------
    snr: float
    the average signal-to-noise ratio
    in the considered interval

    """

    # average signal to noise ratio
    snr = (data / data_err).mean()

    return snr


def get_predicted_flux(dt, a, b, c):
    """Compute the expected flux, in the interval dt,
     using fitted parameters a, b ,c

    Parameters
    ----------
    dt: np.array
    time in days (relative to the initial time t0)
    a:float
    parameter (timescale) of the sigmoid function
    b:float
    parameter (temporal) of the sigmoid function
    c:float
    parameter (height) of the sigmoid function

    Returns
    -------
    predicted_flux: np.array
    with predicted flux based on the fitted values a, b, c

    """

    predicted_array = fsigmoid(dt, a, b, c)
    predicted_flux = pd.DataFrame(predicted_array)\
        .round(decimals=4)\
        .replace(0.0, 0.00001)[0]\
        .values

    return predicted_flux


def get_data_to_export(data_full, data_rising):
    """Export new set of data containing contigous
    rising data

    Parameters
    ----------
    data_full: pandas DataFrame
    full dataset, containing flux, 'MJD' and other labels
    data_rising: pandas DataFrame
    rising dataset, containing flux only

    Returns
    -------
    to_export: pandas DataFrame
    rising dataset, containing flux, 'MJD' and other labels

    """

    to_export = data_full.loc[
        data_full['MJD'].isin(data_rising.index.values)].dropna()

    # set field to zero
    to_export['FIELD'] = 0

    return to_export


def get_train_test(percent_train):
    """Randomly choose test or train label

    Parameters
    ----------
    percent_train: float
    the fraction of the training set
    with respect to the full (training + test) set

    Returns
    -------
    sample: string
    'test' or 'train' set

    """

    rando = random.uniform(0, 1)

    if(rando < percent_train):
        sample = 'train'
    else:
        sample = 'test'

    return sample


def average_intraday_data(df_intra):
    """Average over intraday data points

     Parameters
     ----------
     df_intra: pd.DataFrame
        containing the history of the flux
        with intraday data

     Returns
     -------
     df_average: pd.DataFrame
        containing only daily data
    """

    df_average = df_intra.copy()
    df_average['MJD'] = df_average['MJD'].apply(
        lambda x: np.around(x, decimals=0))
    df_average = df_average.groupby('MJD').mean()
    df_average['MJD'] = df_average.index.values

    return df_average


def get_max_fluxcal(data, list_filters):
    """get the maximum value of FLUXCAL for a
        given alert

    Parameters:
    -----------
    data: pd.DataFrame
        Light curve data. Required key: "FLUXCAL"
    list_filters: list
        List of filters to consider.

    Returns:
    --------
    max_fluxcal: float

    """

    local_max_fluxcal = []
    for i in list_filters:

        data_tmp = filter_data(data, i)
        local_max_fluxcal.append(data_tmp['FLUXCAL'].max())

    max_fluxcal = np.nanmax(local_max_fluxcal)

    return max_fluxcal

def get_sigmoid_features_elasticc(data_all: pd.DataFrame, 
                                  min_data_points=3, 
                                  list_filters=['g', 'r']):
    """Compute the features needed for the Random Forest classification based
    on the sigmoid model.

    Parameters
    ----------
    data_all: pd.DataFrame
        Pandas DataFrame with at least ['MJD', 'BAND', 'FLUXCAL', 'FLUXCALERR']
        as columns.
    min_data_points: int (optional)
        Minimum data points necessary to proceed.
        Default is 3.
    list_filters: list (optional)
        List of filters to considered.
        Default is ['g', 'r'].

    Returns
    -------
    out: list of floats
        List of features, ordered by filter bands:
        [a['X'], b['X'], c['X'], snratio['X'], chisq['X'], nrise['X'] 
        for X in list_filters]
    """
    # lower bound on flux
    low_bound = 0

    # width of the ewma window
    ewma_window = 3

    # N min rising data points
    min_rising_points = 1

    # cutoff on fluxcal
    cutoff_max = 100.

    # features for different filters
    a = {}
    b = {}
    c = {}
    snratio = {}
    chisq = {}
    nrise = {}

    for i in list_filters:
        # select filter
        data_tmp = filter_data(data_all, i)
        
        max_fluxcal = max(data_tmp['FLUXCAL'])
        mjd = data_temp['MJD'].values
        
        if(max_fluxcal > cutoff_max) and \
            max_fluxcal == data_temp['FLUXCAL'].values[np.argsort(mjd)[-1]]:

            # average over intraday data points
            data_tmp_avg = average_intraday_data(data_tmp)
            
            # check data have minimum number of data points
            if len(data_mjd['FLUXCAL'].values) >= min_data_points:
            
                # compute the derivative
                deriv_ewma = get_ewma_derivative(data_mjd['FLUXCAL'],
                                                 ewma_window)
                # mask data with negative part
                data_masked = data_mjd.mask(deriv_ewma < 0)
                
                # get longest raising sequence
                rising_data = data_masked.dropna()

                # only calculate if alert is still rising
                if len(rising_data) >= min_rising_points and \
                   max(rising_data['MJD']) == max(data_mjd['MJD']):

                    # focus on flux
                    rising_time = rising_data['FLUXCAL'].index.values
                    rising_flux = rising_data['FLUXCAL'].values
                    rising_flux_err = rising_data['FLUXCALERR'].values

                    # compute signal to noise ratio
                    snratio[i] = get_sn_ratio(rising_flux, rising_flux_err)

                    # get N rising points
                    nrise[i] = len(rising_flux)

                    dt = delta_t(rising_time)

                    # perform sigmoid fit
                    [a[i], b[i], c[i]] = fit_sigmoid(dt, rising_flux)

                    # predicted flux with fit parameters
                    pred_flux = get_predicted_flux(dt, a[i], b[i], c[i])

                    # compute mse
                    chisq[i] = compute_mse(rising_flux, pred_flux)

                else:
                    # if rising flux has less than min_rising_points
                    [a[i], b[i], c[i], snratio[i], chisq[i], nrise[i]] = \
                        get_fake_results(i)
            else:
                # if data points not enough
                [a[i], b[i], c[i], snratio[i], chisq[i], nrise[i]] = \
                    get_fake_results(i)
    else:
        for i in list_filters:
            # if max fluxcal lower than threshol
            [a[i], b[i], c[i], snratio[i], chisq[i], nrise[i]] = \
                get_fake_results(i)

    result = []
    for f in list_filters:
        result.append(a[f])
        result.append(b[f])
        result.append(c[f])
        result.append(snratio[f])
        result.append(chisq[f])
        result.append(nrise[f])

    return result
                                            
def get_sigmoid_features_elasticc_perfilter(data_all: pd.DataFrame, 
                                  min_data_points=3, 
                                  list_filters=['g', 'r']):
    """Compute the features needed for the Random Forest classification based
    on the sigmoid model.

    Parameters
    ----------
    data_all: pd.DataFrame
        Pandas DataFrame with at least ['MJD', 'BAND', 'FLUXCAL', 'FLUXCALERR']
        as columns.
    min_data_points: int (optional)
        Minimum data points necessary to proceed.
        Default is 3.
    list_filters: list (optional)
        List of filters to considered.
        Default is ['g', 'r'].

    Returns
    -------
    out: list of floats
        List of features, ordered by filter bands:
        [a['X'], b['X'], c['X'], snratio['X'], chisq['X'], nrise['X'] 
        for X in list_filters]
    """
    # width of the ewma window
    ewma_window = 3

    # N min rising data points
    min_rising_points = 1

    # cutoff on fluxcal
    cutoff_max = 100.

    # features for different filters
    a = {}
    b = {}
    c = {}
    snratio = {}
    chisq = {}
    nrise = {}

    # get maximum flux
    max_fluxcal = get_max_fluxcal(data_all, list_filters)
    
    if max_fluxcal > cutoff_max:
    
        for i in list_filters:
            
            # select filter
            data_tmp = filter_data(data_all, i)
            
            # determine lower bound for fluxcal at 10% max fluxcal for this filter
            if data_tmp.shape[0] > 0:
                flux_low_bound = -0.1 * max(data_tmp['FLUXCAL'].values)
                
                # remove values below lower bound
                data_mjd = mask_negative_data(data_tmp, flux_low_bound)

                # average over intraday data points
                data_tmp_avg = average_intraday_data(data_mjd)  

                # check if minimum number of points is respected
                if data_tmp_avg.shape[0] >= min_data_points:
                    # compute the derivative
                    deriv_ewma = get_ewma_derivative(data_tmp_avg['FLUXCAL'],
                                                     ewma_window)
            
                    # mask data with negative part
                    data_masked = data_tmp_avg.mask(deriv_ewma < 0)
                
                    # get longest raising sequence
                    rising_data = data_masked.dropna()
                    
                    # check data have minimum number of data points and if
                    # the alert is still rising
                    if rising_data.shape[0] >= min_rising_points and  \
                        max(rising_data['MJD']) == max(data_tmp_avg['MJD']):

                        # focus on flux
                        rising_time = rising_data['MJD'].index.values
                        rising_flux = rising_data['FLUXCAL'].values
                        rising_flux_err = rising_data['FLUXCALERR'].values

                        # compute signal to noise ratio
                        snratio[i] = get_sn_ratio(rising_flux, rising_flux_err)

                        # get N rising points
                        nrise[i] = len(rising_flux)

                        dt = delta_t(rising_time)

                        # perform sigmoid fit
                        [a[i], b[i], c[i]] = fit_sigmoid(dt, rising_flux)

                        # predicted flux with fit parameters
                        pred_flux = get_predicted_flux(dt, a[i], b[i], c[i])

                        # compute mse
                        chisq[i] = compute_mse(rising_flux, pred_flux)
  
                    else:
                        # if maximum is not last alert or not enough rising points
                        [a[i], b[i], c[i], snratio[i], chisq[i], nrise[i]] = \
                           get_fake_results(i)
                else:
                    # if not enough points for fitting
                    [a[i], b[i], c[i], snratio[i], chisq[i], nrise[i]] = \
                       get_fake_results(i)
            else:
                # if maximum or no points in filter
                [a[i], b[i], c[i], snratio[i], chisq[i], nrise[i]] = \
                  get_fake_results(i)
    else:  
        for i in list_filters:
            # if max fluxcal lower than threshol
            [a[i], b[i], c[i], snratio[i], chisq[i], nrise[i]] = \
                get_fake_results(i)

    result = []
    for f in list_filters:
        result.append(a[f])
        result.append(b[f])
        result.append(c[f])
        result.append(snratio[f])
        result.append(chisq[f])
        result.append(nrise[f])

    return result
 

def get_sigmoid_features_dev(data_all: pd.DataFrame, ewma_window=3, 
                             min_rising_points=2, min_data_points=4,
                             rising_criteria='ewma'):
    """Compute the features needed for the Random Forest classification based
    on the sigmoid model.

    Parameters
    ----------
    data_all: pd.DataFrame
        Pandas DataFrame with at least ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
        as columns.
    ewma_window: int (optional)
        Width of the ewma window. Default is 3.
    min_rising_points: int (optional)
        Minimum number of rising points. Default is 2.
    min_data_points: int (optional)
        Minimum number of data points. Default is 4.
    rising_criteria: str (optional)
        Criteria for defining rising points. Options are 'diff' or 'ewma'.
        Default is 'ewma'.

    Returns
    -------
    out: list of floats
        List of features, ordered by filter bands:
        [a['g'], b['g'], c['g'], snratio['g'], mse['g'], nrise['g'],
         a['r'], b['r'], c['r'], snratio['r'], mse['r'], nrise['r']]

    """
    # lower bound on flux
    low_bound = -10

    list_filters = ['g', 'r']

    # features for different filters
    a = {}
    b = {}
    c = {}
    snratio = {}
    mse = {}
    nrise = {}

    for i in list_filters:
        # select filter
        data_tmp = filter_data(data_all[columns_to_keep], i)
        # average over intraday data points
        data_tmp_avg = average_intraday_data(data_tmp)
        # mask negative flux below low bound
        data_mjd = mask_negative_data(data_tmp_avg, low_bound)

        # check minimum number of points per filter
        if len(data_mjd['FLUXCAL'].values) >= min_data_points:

            if rising_criteria == 'ewma':
                # compute the derivative
                rising_c = get_ewma_derivative(data_mjd['FLUXCAL'], ewma_window)
            elif rising_criteria == 'diff':
                rising_c = get_diff(data_mjd['FLUXCAL'])
                      
            # mask data with negative part
            data_masked = data_mjd.mask(rising_c < 0)
                  
            # get longest raising sequence
            rising_data = data_masked.dropna()

            # at least three points (needed for the sigmoid fit) and all points on the rise
            if(len(rising_data) >= min_rising_points) and len(rising_data) == len(data_mjd):

                # focus on flux
                rising_time = rising_data['FLUXCAL'].index.values
                rising_flux = rising_data['FLUXCAL'].values
                rising_flux_err = rising_data['FLUXCALERR'].values

                # compute signal to noise ratio
                snratio[i] = get_sn_ratio(rising_flux, rising_flux_err)

                # get N rising points
                nrise[i] = len(rising_flux)

                dt = delta_t(rising_time)

                # perform sigmoid fit
                [a[i], b[i], c[i]] = fit_sigmoid(dt, rising_flux)

                # predicted flux with fit parameters
                pred_flux = get_predicted_flux(dt, a[i], b[i], c[i])

                # compute mse
                mse[i] = compute_mse(rising_flux/sum(rising_flux),
                                     pred_flux/sum(pred_flux))

            else:
                # if rising flux has less than three
                [a[i], b[i], c[i], snratio[i], mse[i], nrise[i]] = \
                    get_fake_results(i)
        else:
            # if data points not enough
            [a[i], b[i], c[i], snratio[i], mse[i], nrise[i]] = \
                get_fake_results(i)

    return [
        a['g'], b['g'], c['g'], snratio['g'], mse['g'], nrise['g'],
        a['r'], b['r'], c['r'], snratio['r'], mse['r'], nrise['r']
    ]


def main():
    return None


if __name__ == '__main__':
    main()
