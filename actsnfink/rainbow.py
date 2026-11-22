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


def filter_data_rainbow(mjd, flt, flux, fluxerr=False,
                min_data_points=7,
                list_filters=['g','r'],
                low_bound=-10,
                return_binned_lc=False):
    """Filter only rising alerts for Rainbow fit.

    Parameters
    ----------
    mjd: np.array
        observation time
    flt: np.array
        filter
    flux: np.array
        observed flux
    fluxerr: np.array or bool (optional)
        observed flux error. If False do not take errors into account
        when binning light curve. Default is False.
    min_data_points: int (optional)
        Minimum number of data points in all filters. Default is 7.
    list_filters: list (optional)
        List of filters to consider. Default is ['g', 'r'].
    low_bound: float (optional)
        Lower bound of FLUXCAL to consider. Default is -10.
    return_binned_lc: bool (optional)
        If True, return the binned light curve. Default is False.

    Returns
    -------
    filter_flags: dict
        Dictionary if light curve survived selection cuts.
    lc_binned: pd.DataFrame (optional)
        Binned light curve, only returned if return_binned_lc is True.
    """
    
    
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    # stored binned lc
    lc_binned = []
    
    if is_sorted(mjd):

        # flags if filter survived selection cuts
        filter_flags = dict([[item, False] for item in list_filters])

        if mjd.shape[0] >= min_data_points:
    
            for i in list_filters:
                filter_flag = flt == i

                if sum(filter_flag) > 0:            
                    # mask negative flux below low bound
                    flux_flag = flux >= low_bound
            
                    final_flag = np.logical_and(filter_flag, flux_flag)
    
                    # select filter
                    flux_filter = flux[final_flag]

                    lc = pd.DataFrame()
                    lc['FLUXCAL'] = flux_filter
                    lc['MJD'] = mjd[final_flag]

                    # decide if errors will be taken into account
                    unc = np.bool(fluxerr).all()
                    if unc:
                        lc['FLUXCALERR'] = fluxerr[final_flag]
                
                    # check if it is rising --- THIS SHOULD BE REPLACE BY SOMETHING BETTER
                    avg_data = average_intraday_data(lc, unc=unc)
                    if len(avg_data) > 0:
                        filter_flags[i] = is_sorted(avg_data['FLUXCAL'].values)
                        if return_binned_lc:
                            avg_data['FLT'] = i
                            lc_binned.append(avg_data)
                    else:
                        filter_flags[i] = False
                else:
                        filter_flags[i] = False
        else:
            for i in list_filters:
                filter_flags[i] = False
            
    else:
        raise ValueError('MJD is not sorted!')

    if return_binned_lc == True:
        if len(lc_binned) > 0:
            lc_out = pd.concat(lc_binned, ignore_index=True)
        else:
            lc_out = None
            
        return filter_flags, lc_out
    else:
        return filter_flags


def fit_rainbow(mjd, flt, flux, fluxerr, 
                band_wave_aa={"g": 4770.0, "r": 6231.0, "i": 7625.0},
                with_baseline=False, 
                min_data_points=7,
                list_filters=['g','r'],
                nrising_filters=2,
                low_bound=-10,
                use_binned_lc=False):
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
    nrising_filters: int (optional)
        Number of filters going required for feature extraction.
    low_bound: float (optional)
        Lower bound of FLUXCAL to consider. Default is -10.
    use_binned_lc: bool (optional)
        If True, use binned lc for rainbow fit. Default is False.

    Returns
    -------
    values: list
        list of best-fit parameters for the Rainbow model.
    errors: list
        list of uncertainty in each parameter.
    npoints: int
        number of points used to fit
    """

    indx = np.argsort(mjd)
    if use_binned_lc:
        filter_flag, lc_binned =  filter_data_rainbow(mjd[indx], flt[indx], 
                                                      flux[indx], 
                                                      fluxerr=fluxerr[indx],
                                                      min_data_points=min_data_points,
                                                      list_filters=list_filters,
                                                      low_bound=low_bound,
                                                      return_binned_lc=use_binned_lc)
    else:
        filter_flag =  filter_data_rainbow(mjd[indx], flt[indx], 
                                                      flux[indx], 
                                                      fluxerr=fluxerr[indx],
                                                      min_data_points=min_data_points,
                                                      list_filters=list_filters,
                                                      low_bound=low_bound)

    # check if required number of filters survived
    if sum(filter_flag.values()) >= nrising_filters:
        if use_binned_lc:
            lc_max = max(lc_binned['FLUXCAL'])

            lc_norm = pd.DataFrame({'mjd': lc_binned['MJD'],
                                   'flux': lc_binned['FLUXCAL']/lc_max,
                                   'fluxerr': lc_binned['FLUXCALERR']/lc_max,
                                   'band': lc_binned['FLT']})

            # generate small differences in MJD which should be unique
            if np.unique(lc_norm['mjd']).shape[0] < lc_norm.shape[0]:
              
                mjd_diff = []
                lc_binned_sorted = deepcopy(lc_norm.sort_values(by='mjd'))
                for i in range(lc_binned_sorted.shape[0] - 1):
                    if lc_binned_sorted.iloc[i]['mjd'] == lc_binned_sorted.iloc[i+1]['mjd']:
                        mjd_diff.append(lc_binned_sorted.iloc[i]['mjd'] + 0.01 * np.random.random())
                    else:
                        mjd_diff.append(lc_binned_sorted.iloc[i]['mjd'])
                mjd_diff.append(lc_binned_sorted.iloc[lc_binned_sorted.shape[0] - 1]['mjd'])
                lc_binned_sorted['mjd'] = mjd_diff
                lc_binned_sorted.sort_values(by='mjd', inplace=True)

                # order again
                mjd = lc_binned_sorted['mjd']
                flux_norm = lc_binned_sorted['flux']
                fluxerr_norm = lc_binned_sorted['fluxerr']
                flt = lc_binned_sorted['band']

            else:
                lc_sorted = lc_norm.sort_values(by='mjd')
                flux_norm = lc_sorted['flux']
                fluxerr_norm = lc_sorted['fluxerr']
                mjd = lc_sorted['mjd']
                flt = lc_sorted['band']
                
        else:        
            # normalize light curve
            lc_max = max(flux)
            flux_norm = flux/lc_max
            fluxerr_norm = fluxerr/lc_max
        
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