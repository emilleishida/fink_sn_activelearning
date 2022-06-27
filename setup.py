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

import setuptools

setuptools.setup(
    name='actsnfink',
    version='0.1',
    packages=setuptools.find_packages(),
    py_modules=['classifier_sigmoid',
                'early_sn_classifier',
                'sigmoid'],
    scripts=['actsnfink/scripts/run_loop.py'],
    url='https://github.com/emilleishida/fink_sn_activelearning',
    license='MIT',
    author='Emille E. O.Ishida',
    author_email='emille.ishida@clermont.in2p3.fr',
    description='Fink Early SN classifier using Active Learning'
)
