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

from light_curve.light_curve_py import RainbowFit
from actsnfink.classifier_sigmoid import average_intraday_data

__all__ = ['extract_history', 'extract_field', 
           'filter_data_rainbow', 'fit_rainbow']

columns_to_keep = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']

def extract_history(history_list: list, field: str) -> list:
    """Extract the historical measurements contained in the alerts
    for the parameter `field`.

    Parameters
    ----------
    history_list: list of dict
        List of dictionary from alert[history].
    field: str
        The field name for which you want to extract the data. It must be
        a key of elements of history_list
    
    Returns
    ----------
    measurement: list
        List of all the `field` measurements contained in the alerts.
    """
    if history_list is None:
        return []
    try:
        measurement = [obs[field] for obs in history_list]
    except KeyError:
        print('{} not in history data'.format(field))
        measurement = []

    return measurement

def extract_field(alert: dict, category: str, field: str) -> np.array:
    """ Concatenate current and historical observation data for a given field.
    
    Parameters
    ----------
    alert: dict
        Dictionnary containing alert data
    category: str
        prvDiaSources or prvDiaForcedSources
    field: str
        Name of the field to extract.
    
    Returns
    ----------
    data: np.array
        List containing previous measurements and current measurement at the
        end. If `field` is not in the category, data will be
        [alert['diaSource'][field]].
    """
    data = np.concatenate(
        [
            extract_history(alert[category], field),
            [alert["diaSource"][field]]
        ]
    )
    return data


def filter_data_rainbow(mjd, flt, flux,
                min_data_points=7,
                list_filters=['g','r'],
                low_bound=-10):
    """Filter only rising alerts for Rainbow fit.

    Parameters
    ----------
    data_all: pd.DataFrame
        Pandas DataFrame with at least ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
        as columns.
    min_data_points: int (optional)
        Minimum number of data points in all filters. Default is 7.
    list_filters: list (optional)
        List of filters to consider. Default is ['g', 'r'].
    low_bound: float (optional)
        Lower bound of FLUXCAL to consider. Default is -10.

    Returns
    -------
    filter_flags: dict
        Dictionary if light curve survived selection cuts.
    """
    
    
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    
    if is_sorted(mjd):

        # flags if filter survived selection cuts
        filter_flags = dict([[item, False] for item in list_filters])

        if mjd.shape[0] >= min_data_points:
    
            for i in list_filters:
                filter_flag = flt == i
            
                # mask negative flux below low bound
                flux_flag = flux >= low_bound
            
                final_flag = np.logical_and(filter_flag, flux_flag)
    
                # select filter
                flux_filter = flux[final_flag]

                lc = pd.DataFrame()
                lc['FLUXCAL'] = flux_filter
                lc['MJD'] = mjd[final_flag]
                
                # check if it is rising
                avg_data = average_intraday_data(lc)
                flux_sorted = is_sorted(avg_data['FLUXCAL'].values)

                if flux_sorted:
                    filter_flags[i] = True
                else:
                    filter_flags[i] = False
        else:
            for i in list_filters:
                filter_flags[i] = False
            
    else:
        raise ValueError('MJD is not sorted!')
        
    return filter_flags


def fit_rainbow(mjd, flt, flux, fluxerr, 
                band_wave_aa={"g": 4770.0, "r": 6231.0, "i": 7625.0},
                with_baseline=False, 
                min_data_points=7,
                list_filters=['g','r'],
                low_bound=-10):
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
    values: list
        list of best-fit parameters for the Rainbow model.
    errors: list
        list of uncertainty in each parameter.
    npoints: int
        number of points used to fit
    """
    
    filter_flag =  filter_data_rainbow(mjd, flt, flux, 
                     min_data_points=min_data_points,
                     list_filters=list_filters,
                     low_bound=low_bound)

    # at least one filter survived
    if sum(filter_flag.values()) > 2:
        # normalize light curve    

        lc_max = max(flux)
        flux_norm = flux/lc_max
        fluxerr_norm = fluxerr/lc_max
    
        npoints = flux_norm.shape[0]
    
        # extract features
        feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline=with_baseline,
                                           bolometric='sigmoid', temperature='sigmoid')
        
        try:
            values = feature(mjd, flux_norm, 
                             sigma=fluxerr_norm, band=flt)
            
            res = list(values) + [lc_max]
            
        except RuntimeError:
            res = [0 for i in range(8)]

        return res
    
    else:
        return [0 for i in range(8)]
