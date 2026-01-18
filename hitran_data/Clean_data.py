#%%
import pandas as pd
import numpy as np

df = pd.read_csv("H2O/6278f3b7.txt")
#%% md
# 
#%%
print(df.columns)
#%%
import os
from glob import iglob

rootdir_glob = '/Users/elizabeth/Desktop/line_broadening/hitran_data/**/*' # Note the added asterisks
# This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-3:] == "txt" if "readme" not in f]
#%%
db = {}

for f in file_list:
    i = f[53:-13]
    db[i] = pd.read_csv(f)

#%%
params = db['H2O'].columns

for key, value in db.items():
    print(value.columns)
    value = value.iloc[:, 0:15]
    print(value.columns)
#%%
from itertools import groupby

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)
#%%
for key, value in db.items():
    x = value.groupby(['statep', 'statepp']).agg(list)
    for index, row in x.iterrows():
        y = row['gamma_air']
        if not all_equal(y):
            print(row)
#%%
