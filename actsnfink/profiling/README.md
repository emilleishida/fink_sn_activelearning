# Profiling module 
This directory contains a script to profile the performance of some functions. 

## Requirements 

Install the profoling tool:
```
source venv/bin/activate
pip install -r requirements.txt
pip install . # to install package actsnfink

```

## Running the profiler
Run (and display) the profiling script with:
```

kernprof -l -v actsnfink/profiling/profile.py 
```

This will show the execution time for each line of the profiled functions : 

```python
File: ../fink_sn_activelearning/actsnfink/classifier_sigmoid.py
Function: get_sigmoid_features_dev at line 345

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   345                                           @profile
   346                                           def get_sigmoid_features_dev(data_all: pd.DataFrame, ewma_window=3, 
   347                                                                        min_rising_points=2, min_data_points=4,
   348                                                                        rising_criteria='ewma'):
   349                                               """Compute the features needed for the Random Forest classification based
   350                                               on the sigmoid model.
   ...
   373                                           
   374                                               """
   375                                               # lower bound on flux
   376         1          0.9      0.9      0.0      low_bound = -10
   377                                           
   378         1          0.5      0.5      0.0      list_filters = ['g', 'r']
   379                                           
   380                                               # features for different filters
   381         1          0.7      0.7      0.0      a = {}
   382         1          0.4      0.4      0.0      b = {}
   383         1          0.2      0.2      0.0      c = {}
   384         1          0.5      0.5      0.0      snratio = {}
   385         1          0.3      0.3      0.0      mse = {}
   386         1          0.2      0.2      0.0      nrise = {}
   387                                           
   388         3          1.8      0.6      0.0      for i in list_filters:
   389                                                   # select filter
   390         2       4196.6   2098.3     29.3          data_tmp = filter_data(data_all[columns_to_keep], i)
   391                                                   # average over intraday data points
   392         2       8208.4   4104.2     57.3          data_tmp_avg = average_intraday_data(data_tmp)
   393                                                   # mask negative flux below low bound
   394         2       1769.4    884.7     12.3          data_mjd = mask_negative_data(data_tmp_avg, low_bound)
```

What matters first is the column `% Time` which indicates the percentage of time
spent per call. In this example above, 57.3 % is spent in calling ` average_intraday_data(data_tmp)`
which would be the target to optimize if we want to improve the performances.

Another important column is `Hits`, that is the number of time an instruction has been done.
In this example, we  checked the loop condition 3 times. 


## TODO: 
Perform profiling (or not) for the other functions of the **actsnfink** module in the **profile.py** script, with the aim of finding bottlenecks and optimizations.