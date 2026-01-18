#%%
import pandas as pd
import numpy as np
import os
from glob import iglob

rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/hitran_data/**/*' # Note the added asterisks
# This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-10:] == "/1_iso.csv" if "readme" not in f]
#%%
db = {}

for f in file_list:
    i = f[76:-10]
    db[i] = pd.read_csv(f)
#%%
del db["HONO_prediction"]
#%%
print(len(db.keys()))
#%%
for key, value in db.items():
    #print(key)
    #print(len(value))
    if key == 'GeH4':
        print(key)
    
#%%
keys = []
total_lines = []
air_lines = []
self_lines = []
H2_lines = []
He_lines = []
Co2_lines = []
H2o_lines = []

air_good = []
self_good = []
H2_good = []
He_good = []
CO2_good = []
H2O_good = []



#%%

print(db['H2']['gamma_H2O'])
#%%
for key, value in db.items():
    if 'gamma_H2' not in value.columns:
        db[key]['gamma_H2'] = '#'
        db[key]['gamma_He'] = '#'
        db[key]['gamma_H2O'] = '#'
        db[key]['gamma_CO2'] = '#'
#%%
for key, value in db.items():
    keys.append(key)
    total_lines.append(len(value))
    

    Gair = value['gamma_air']
    if '#' in Gair.value_counts():
        air_lines.append(len(Gair) - Gair.value_counts()['#'])
    else:
        air_lines.append(len(Gair))
    
    air_good.append(value['gamma_air-err'].value_counts())

    Gself = value['gamma_self']
    if '#' in Gself.value_counts():
        self_lines.append(len(Gself) - Gself.value_counts()['#'])
    else:
        self_lines.append(len(Gself))

    self_good.append(value['gamma_self-err'].value_counts())

    Gh2 = value['gamma_H2']
    if '#' in Gh2.value_counts():
        H2_lines.append(len(Gh2) - Gh2.value_counts()['#'])
    else:
        H2_lines.append(len(Gh2))
        
    #air_good.append(value['gamma_air-err'])


    Ghe = value['gamma_He']
    if '#' in Ghe.value_counts():
        He_lines.append(len(Ghe) - Ghe.value_counts()['#'])
    else:
        He_lines.append(len(Ghe))

    #air_good.append(value['gamma_air-err'])

    Gh2o = value['gamma_H2O']
    if '#' in Gh2o.value_counts():
        H2o_lines.append(len(Gh2o) - Gh2o.value_counts()['#'])
    else:
        H2o_lines.append(len(Gh2o))

    #air_good.append(value['gamma_air-err'])

    Gco2 = value['gamma_CO2']
    if '#' in Gco2.value_counts():
        Co2_lines.append(len(Gco2) - Gco2.value_counts()['#'])
    else:
        Co2_lines.append(len(Gco2))
        
    #air_good.append(value['gamma_air-err'])
#%%
print(keys)
print(total_lines)
print(air_lines)
print(self_lines)
print(H2_lines)
print(He_lines)
print(Co2_lines)
print(H2o_lines)
#%%
print(air_good)
print(self_good)
#%%
airg = []
selfg = []
#%%
for value in air_good:
    airg.append(value[value.index > 2].sum()/value.sum()*100)
for value in self_good:
    selfg.append(value[value.index > 2].sum()/value.sum()*100)
#%%
from operator import truediv

air_prop = list(map(truediv, air_lines, total_lines))
He_prop = list(map(truediv, He_lines, total_lines))
H2_prop = list(map(truediv, H2_lines, total_lines))
Co2_prop = list(map(truediv, Co2_lines, total_lines))
H2o_prop = list(map(truediv, H2o_lines, total_lines))
self_prop = list(map(truediv, self_lines, total_lines))
#%%
table = pd.DataFrame(list(zip(total_lines, air_lines, air_prop, airg, He_lines, He_prop, H2_lines, H2_prop, Co2_lines, Co2_prop, H2o_lines, H2o_prop, self_lines, self_prop, selfg)), index =keys,
                                              columns =['total_lines', 'air_lines', 'air_prop', 'percentage of good air data', 'He_lines', 'He_prop', 'H2_lines', 'H2_prop', 'Co2_lines', 'Co2_prop', 'H2o_lines', 'H2o_prop', 'self_lines', 'self_prop', 'percentage of good self broadening data'])
#%%
print(table)
#%%
table_clean = table.drop(['self_lines', 'air_lines', 'He_lines', 'H2_lines', 'Co2_lines', 'H2o_lines'], axis=1)
#%%
print(table_clean)
table_clean
#%%
table_clean.astype(int)
#%%
table_clean[(table_clean!=0)][(table_clean!=1)]
#%%
table_clean.to_csv("table.csv")
#%%

#%%

#%%
