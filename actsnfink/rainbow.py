# Copyright 2023 Fink Software
# Author: Emille E. O. Ishida
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


from actsnfink.classifier_sigmoid import *
from light_curve.light_curve_py import RainbowFit

__all__ = ['filter_data_rainbow', 'fit_rainbow', 'fit_rainbow_dataset']

columns_to_keep = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']


def filter_data_rainbow(data_all: pd.DataFrame, ewma_window=3, 
                min_rising_points=1, min_data_points=7,
                rising_criteria='ewma', list_filters=['g','r'],
                low_bound=-10):
    """Filter only rising alerts for Rainbow fit.

    Parameters
    ----------
    data_all: pd.DataFrame
        Pandas DataFrame with at least ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
        as columns.
    ewma_window: int (optional)
        Width of the ewma window. Default is 3.
    min_rising_points: int (optional)
        Minimum number of rising points per filter. Default is 1.
    min_data_points: int (optional)
        Minimum number of data points in all filters. Default is 7.
    rising_criteria: str (optional)
        Criteria for defining rising points. Options are 'diff' or 'ewma'.
        Default is 'ewma'.
    list_filters: list (optional)
        List of filters to consider. Default is ['g', 'r'].
    low_bound: float (optional)
        Lower bound of FLUXCAL to consider. Default is -10.

    Returns
    -------
    filter_flags: dict
        Dictionary if light curve survived selection cuts.
    """

    # flags if filter survived selection cuts
    filter_flags = dict([[item, False] for item in list_filters])

    if data_all.shape[0] >= min_data_points:
    
        for i in list_filters:
            # select filter
            data_tmp = filter_data(data_all[columns_to_keep], i)
            # average over intraday data points
            data_tmp_avg = average_intraday_data(data_tmp)
            # mask negative flux below low bound
            data_mjd = mask_negative_data(data_tmp_avg, low_bound)
  
            # we need at least 1 point in each filter
            if len(data_mjd['FLUXCAL'].values) > 0:
                if rising_criteria == 'ewma':
                    # compute the derivative
                    rising_c = get_ewma_derivative(data_mjd['FLUXCAL'], ewma_window)
                elif rising_criteria == 'diff':
                    rising_c = get_diff(data_mjd['FLUXCAL'])
                      
                # mask data with negative part
                data_masked = data_mjd.mask(rising_c < 0)
                  
                # get longest raising sequence
                rising_data = data_masked.dropna()

                # count points on the rise
                if(len(rising_data) >= min_rising_points) and len(rising_data) == len(data_mjd):
                    filter_flags[i] = True
                else:
                    filter_flags[i] = False
            else:
                filter_flags[i] = False

    return filter_flags


def fit_rainbow(lc: pd.DataFrame, 
                band_wave_aa={"g": 4770.0, "r": 6231.0, "i": 7625.0},
                with_baseline=False, low_bound=-10):
    """Use Rainbow to fit light curve.

    Parameters
    ----------
    lc: pd.DataFrame
        Pandas DataFrame with at least ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
        as columns.        
    band_wave_aa: dict (optional)
        Dictionary with effective wavelength for each filter. 
        Default is for ZTF: {"g": 4770.0, "r": 6231.0, "i": 7625.0} 
    with_baseline: bool (optional)
        Baseline to be considered. Default is False (baseline 0).
    low_bound: float (optional)
        Lower bound of FLUXCAL to consider. Default is -10.

    Returns
    -------
    features: list
        list of best-fit parameters for the Rainbow model.
    """

    # normalize light curve
    lc2 = mask_negative_data(lc, low_bound)
    
    lc_max = max(lc2[lc2['FLT'] == 'r']['FLUXCAL'])
    indx_max = list(lc2['FLUXCAL'].values).index(lc_max)
    lc3 = pd.DataFrame()
    lc3['FLUXCAL'] = lc2['FLUXCAL'].values/lc_max
    lc3['FLUXCALERR'] = lc2['FLUXCALERR'].values/lc_max
    lc3['MJD'] = lc2['MJD'].values
    lc3['FLT'] = lc2['FLT'].values

    lc3['MJD'] = np.where(lc3['MJD'].duplicated(keep=False), 
                      lc3['MJD'] + lc3.groupby('MJD').cumcount().add(0.25).astype(float),
                      lc3['MJD'])
    data_use = deepcopy(lc3.sort_values(by=['MJD'], ignore_index=True))
    
    # extract features
    feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline=with_baseline)
    values = feature(data_use['MJD'].values, data_use['FLUXCAL'].values, 
                     sigma=data_use['FLUXCALERR'].values, band=data_use['FLT'].values)

    return values


def fit_rainbow_dataset(data_all: pd.DataFrame, 
                        id_name='id', ewma_window=3, 
                        min_rising_points=1, min_data_points=7,
                        rising_criteria='ewma', list_filters=['g','r'],
                        low_bound=-10, 
                        band_wave_aa={"g": 4770.0, "r": 6231.0, "i": 7625.0},
                        with_baseline=False):
    """Process an entire data set with Rainbow fit.

    Parameters
    ----------
    data_all: pd.DataFrame
        Pandas DataFrame with at least ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
        as columns.
    ewma_window: int (optional)
        Width of the ewma window. Default is 3.
    id_name: str (optional)
        String identifying the column for each object. Default is 'id'.
    min_rising_points: int (optional)
        Minimum number of rising points per filter. Default is 1.
    min_data_points: int (optional)
        Minimum number of data points in all filters. Default is 7.
    rising_criteria: str (optional)
        Criteria for defining rising points. Options are 'diff' or 'ewma'.
        Default is 'ewma'.
    list_filters: list (optional)
        List of filters to consider. Default is ['g', 'r'].
    low_bound: float (optional)
        Lower bound of FLUXCAL to consider. Default is -10.      
    band_wave_aa: dict (optional)
        Dictionary with effective wavelength for each filter. 
        Default is for ZTF: {"g": 4770.0, "r": 6231.0, "i": 7625.0} 
    with_baseline: bool (optional)
       Baseline to be considered. Default is False (baseline 0).

    Returns
    -------
    rainbow_fits: pd.DataFrame
         Ids and Rainbow features for all ids surviving cuts.
    """
    # store results
    results_list = []
    
    # get unique ids
    unique_ids = np.unique(data_all['id'].values)

    for snid in unique_ids:
        lc = data_all[data_all['id'].values == snid]
        flag_surv = deepcopy(filter_data_rainbow(lc, rising_criteria=rising_criteria))
    
        if sum(flag_surv.values()) == 2:
            features = fit_rainbow(lc, band_wave_aa=band_wave_aa,
                                  with_baseline=with_baseline)

            results_line = [snid] + list(features)
            results_list.append(results_line)

    if with_baseline:
        names = [id_name, 't0', 'amplitude', 'rise_time', 
                                       'temperature', 'reduced_chi2', 'baseline1', 'baseline2']
    else:
        names = [id_name, 't0', 'amplitude', 'rise_time', 
                                       'temperature', 'reduced_chi2']
        
    results_pd = pd.DataFrame(results_list, 
                              columns=names)

    return results_pd