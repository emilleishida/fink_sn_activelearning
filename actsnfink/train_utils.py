# Copyright 2024 
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

from actsnclass import DataBase
from copy import deepcopy

import pandas as pd
import numpy as np

def read_samples_sequential_sets(fname_train: str, fname_queried:str,
                                 fname_available: str, n: int, 
                                 meta_names=['id','type'],
                                 use_alertid='diaSourceId',
                                 use_objid='diaObjectId'
                                 ):
    """
    Build DataBase objects using test and training from file.
    This was built for training big data sets quickly.
    It continues the learning loop based on previous ones.

    Parameters
    ----------
    fname_train: str
        Full path to file containing training from previous loop.
    fname_queried: str
        Full path to file containing queries made in previous loop.
    fname_available: str
        Full path containing data to be used for test in this loop.
    meta_names: list of str (optional)
        List of metadata columns. Default is ['id', 'type'].
    use_alertid: str (optional)
        Name of column identifying each alert. Default is 'diaSourceId'.
    use_objid: str (optional)
        Name of column identifying each object. Default is 'diaObjectId'.
    
    Returns
    -------
    DataBase object from ActSNClass
    """

    # read initial train from previous loop
    train_pd = pd.read_csv(fname_train)

    # read queries from previous loop
    queried_pd = pd.read_csv(fname_queried)
    queried_pd.rename(columns={'id':use_alertid}, inplace=True)

    # build new train
    new_train = pd.concat([train_pd, queried_pd[list(train_pd.keys())]], ignore_index=True)

    # build new test sample
    avail_pd = pd.read_parquet(fname_available)

    # remove training objects from target
    flag_train = np.isin(avail_pd[use_objid].values, new_train[use_objid].values)
    avail_pd_test = avail_pd[~flag_train]

    # create new test set
    test_names = np.unique(avail_pd_test[use_objid].values)              # get unique objectId
    ids_test = np.random.choice(test_names, size=n, replace=False)       # get subset of objectId
    flag_test = np.isin(avail_pd_test[use_objid].values, ids_test)       # flag all alerts
    new_test = avail_pd_test[flag_test]                                  # new test formed by alerts

    # build DataBase object
    data = DataBase()

    features_names = list(train_pd.keys())
    for item in meta_names:
        features_names.remove(item)
    
    data.features_names = features_names
    data.metadata_names = meta_names

    data.train_features = new_train[features_names].values
    data.train_metadata = deepcopy(new_train[meta_names])
    data.train_metadata.rename(columns={use_alertid:'id'}, inplace=True)
    train_labels = new_train['type'].values == 'Ia'
    data.labels = train_labels.astype(int)

    data.test_features = new_test[features_names].values
    data.test_metadata = deepcopy(new_test[meta_names])
    data.test_metadata.rename(columns={use_alertid:'id'}, inplace=True)
    # test_labels = new_test['type'].values == 'Ia'
    data.test_labels = data.test_metadata['type'].values

    # mark all as queryable
    data.queryable_ids = data.test_metadata['id'].values

    return data


def main():
    return None
    
if __name__ == '__main__':
    main()