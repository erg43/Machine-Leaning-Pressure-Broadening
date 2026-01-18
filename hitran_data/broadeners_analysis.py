#%%
# import packages
import pandas as pd
import numpy as np
import os
from glob import iglob

#%%
# import data

# absolute path to folder containing data
rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/hitran_data/**/*'
# be selective for data files
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-10:] == "/1_iso.csv" if "readme" not in f]

# read data files, taking the filename from absolute path
db = {}
for f in file_list:
    i = f[76:-10]
    db[i] = pd.read_csv(f)
#%%
ho = db['OH'].loc[2178]
#%%
ho['gamma_air']

#%%
gamma_types = ['gamma_air', 'gamma_self', 'gamma_H2', 'gamma_He', 'gamma_CO2', 'gamma_H2O']

for key, data in db.items():
    print(key)
    for column in data.columns:
        if column in gamma_types:
            print()
            print(column[6:]+' broadening')
            dtypes = data[column].value_counts()
            if '#' in dtypes.index:
                print(str(100 - 100*dtypes['#']/len(data))+f'% of data present')
            else:
                print(f'100% of data present')
                
    print()
    print()
#%%
