import pandas as pd 
import numpy as np
import random 
from actsnfink.classifier_sigmoid import get_fake_df,get_sigmoid_features_dev,get_sigmoid_features_dev_fast

df = get_fake_df('g') 

print("Original version")
l1 = get_sigmoid_features_dev(df)
print("Optimised version")
l2 = get_sigmoid_features_dev_fast(df)

# Just for testing 
assert l1==l2,"Not the same results! "

