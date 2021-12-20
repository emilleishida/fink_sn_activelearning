# <img align="right" src="docs/images/Fink_PrimaryLogo_WEB.png" width="350"> Fink

## Fink: early supernovae Ia classification using active learning

This repository contains code allowing reproducibility of results presented in [Leoni et al., 2021, arxvi:astro-ph/2111.11438](https://arxiv.org/abs/2111.11438)

We list below a general description of each script/notebook. 

The data necessary to reproduce these results are available through [zenodo](https://zenodo.org/record/5645609#.YcD3przMJNg).

- [code/sigmoid.py](https://github.com/emilleishida/fink_sn_activelearning/blob/master/code/sigmoid.py): 
    functions related to the sigmoid feature extraction
    
- [classifier_sigmoid.py](https://github.com/emilleishida/fink_sn_activelearning/blob/master/code/classifier_sigmoid.py): 
    functions related to filtering points on the rise and concatenation with extra features (SNR, npoints, chi2) with sigmoid fit parameters

- [early_sn_classifier.py](https://github.com/emilleishida/fink_sn_activelearning/blob/master/code/early_sn_classifier.py):
    global functions for feature extraction and learning loop. 
    User choices are restricted to lines 462-497.
    In order to generate the first data matrix it is necessary to access the AL_data folder, which is provided in an accompaning file.
    
- [mean_model.ipynb](https://github.com/emilleishida/fink_sn_activelearning/blob/master/code/mean_model.ipynb):
    Extract best performing model from a given query strategy, save pkl file and generate list of alerts used of training.
    
- [plots/](https://github.com/emilleishida/fink_sn_activelearning/tree/master/code/plots):
    Folder of jupyter notebooks for reproducing the plots in Leoni et al., 2021
    
- [LICENSE]():
    MIT License