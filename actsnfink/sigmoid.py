# Copyright 2020-2021 AstroLab Software
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

import numpy as np

from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error

__all__ = ['delta_t', 'fsigmoid', 'errfunc_sigmoid', 'fit_sigmoid', 'compute_mse']


def delta_t(time_index: np.array) -> np.array:
    """ Re-index an index relatively to the first data point.

    Parameters
    ----------
    time_index : np.array

    Returns
    -------
    relative_time : np.array
        time relative to the first
        data point in the dataframe
    """

    relative_time = time_index - time_index[0]

    return relative_time


def fsigmoid(x: np.array, a: float, b: float, c: float) -> np.array:
    """Sigmoid function

    Parameters
    ---------
    x: np.array
    a: float
    b: float
    c: float

    Returns
    -------
    sigmoid: np.array
        fit with a sigmoid function
    """

    sigmoid = c / (1.0 + np.exp(-a * (x - b)))

    return np.array(sigmoid)


def errfunc_sigmoid(params: list, time: np.array, flux: np.array) -> float:
    """ Absolute difference between theoretical and measured flux.

    Parameters
    ----------
    params : list of float
        light curve parameters: (a, b, t0, tfall, trise)
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)

    Returns
    -------
    diff : float
        absolute difference between theoretical and observed flux

    """
    return abs(flux - fsigmoid(time, *params))


def fit_sigmoid(time: np.array, flux: np.array) -> list:
    """ Find best-fit parameters using scipy.least_squares.

    Parameters
    ----------
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)

    Returns
    -------
    result : list of float
        best fit parameter values
    """
    flux = np.asarray(flux)
    t0 = time[flux.argmax()] - time[0]
    if t0 > 0:
        dt = time[flux.argmax()] - time[flux.argmin()]
        slope = (flux.argmax() - flux.argmin()) / dt
    else:
        slope = 1.
    if flux[0] > 0:
        f0 = flux[0]
    else:
        f0 = 0
        
    aguess = slope
    cguess = np.max(flux)

    if f0 != 0 and cguess / f0 != 1.:
        bguess = np.log(cguess / f0 - 1.) / aguess
    else:
        bguess = 1.0

    guess = [aguess, bguess, cguess]
    result = least_squares(errfunc_sigmoid, guess, args=(time, flux))

    return result.x


def compute_mse(f_obs: np.array, f_exp: np.array) -> float:
    """ Compute mean squared error.

    Parameters
    ----------
    f_obs: np.array
        observed data points
    f_exp: np.array
        fitted (predicted) data points

    Returns
    -------
    test_mse: float
        mse between fitted and observed
    """

    test_mse = mean_squared_error(f_obs, f_exp)
    
    return test_mse


def main():
    return None
    
if __name__ == '__main__':
    main()