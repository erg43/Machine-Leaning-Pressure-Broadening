#%%
import pandas as pd
import numpy as np
import os
from glob import iglob

rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/hitran_data/**/*' # Note the added asterisks
# This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-7:] == "raw.txt" if "readme" not in f]
#%%
db = {}

for f in file_list:
    i = f[76:-7]
    db[i] = pd.read_csv(f)
#%%
for key, data in db.items():
    print(key)
    print(len(data['local_iso_id']))
    df_main = data.loc[data['local_iso_id'] == 1]
    print(len(df_main['local_iso_id']))
    print('~~~~~~~~~~~~~~~~~~~~~~~~')
#%%
for key, data in db.items():
    print(key)
    print(data['local_iso_id'].value_counts())
    #df_main = data.loc[data['local_iso_id'] == 1]
    #print(len(df_main['local_iso_id']))
    print('~~~~~~~~~~~~~~~~~~~~~~~~')
#%%
import matplotlib.pyplot as plt
for key, data in db.items():
    isotopes = []
    average_gamma_err = []
    print(key)
    groups = data.groupby('local_iso_id')
    for isotope, info in groups:
        isotopes.append(isotope)
        #print(info['gamma_air-err'].value_counts())
        average_gamma_err.append(np.mean(info['gamma_air-err']))
        #print()
        #print(groups.groups)
    plt.bar(isotopes, average_gamma_err)
    plt.title(key[:-1])
    plt.xlabel('isotope')
    plt.ylabel('average error code')
    plt.show()
    #df_main = data.loc[data['local_iso_id'] == 1]
    #print(len(df_main['local_iso_id']))
    print('~~~~~~~~~~~~~~~~~~~~~~~~')
#%%
for key, data in db.items():
    print(data.columns)
    break
#%%
for key, data in db.items():
    
    new = data["statep"].str.split(";", expand=True)
    data["J"]= data['statep'].str.extract(r'(J=)(.+?);')[1]
    # Dropping old Name columns
    data.drop(columns =['statep'], inplace = True)
    data['statepp'] = data["statepp"]+str(';')
    new = data["statepp"].str.split(";", expand=True)
    data["Jpp"]= data['statepp'].str.extract(r'(J=)(.+?);')[1]
    # Dropping old Name columns
    data.drop(columns =['statepp'], inplace = True)
    data.drop(columns=['gamma_H2', 'n_H2', 'gamma_He', 'n_He', 'gamma_CO2', 'n_CO2',
       'gamma_H2O', 'n_H2O', 'nu-err', 'sw-err', 'gamma_air-err', 'n_air-err',
       'gamma_self-err', 'n_self-err', 'gamma_H2-err', 'n_H2-err',
       'gamma_He-err', 'n_He-err', 'gamma_CO2-err', 'n_CO2-err',
       'gamma_H2O-err', 'n_H2O-err', 'nu-ref', 'sw-ref', 'gamma_air-ref',
       'n_air-ref', 'gamma_self-ref', 'n_self-ref', 'gamma_H2-ref', 'n_H2-ref',
       'gamma_He-ref', 'n_He-ref', 'gamma_CO2-ref', 'n_CO2-ref',
       'gamma_H2O-ref', 'n_H2O-ref'])
    print(key)
    print(len(data))
    data = data.dropna()
    print(len(data))
    data['J'] = data['J'].astype('float')
    data['Jpp'] = data['Jpp'].astype('float')
    branch = data['Jpp'] - data['J']
    data = data.drop(branch[abs(branch) > 2].index)
    branch = data['Jpp'] - data['J']
    
    data['P'] = -data['Jpp'][branch == 1]
    data['Q'] = data['Jpp'][branch == 0]
    data['R'] = data['Jpp'][branch == -1] + 1
    data['O'] = -data['Jpp'][branch == 2]
    data['S'] = data['Jpp'][branch == -2] + 1
    
    data['P'] = data['P'].fillna(0)
    data['Q'] = data['Q'].fillna(0)
    data['R'] = data['R'].fillna(0)
    data['O'] = data['O'].fillna(0)
    data['S'] = data['S'].fillna(0)
    #data = data.fillna(0)
    
    data['M'] = data['P'] + data['Q'] + data['R'] + data['O'] + data['S']
    data = data.drop(columns=['P', 'Q', 'R', 'O', 'S'])

    db[key] = data
#%%
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
for key, data in db.items():
    isotopes = []
    average_gamma_err = []
    print(key)
    groups = data.groupby('local_iso_id')
    figure(figsize=((15, 7)), dpi=300)
    fig = plt.figure(1)
    plt.rc('font', size=14)
    plt.rcParams['axes.facecolor'] = 'white'
    for isotope, info in groups:
        average_gamma_err.append(np.mean(info['gamma_air-err']))
        info.sample(frac=1)
        plt.scatter(info['M'][:1000], info['gamma_air'][:1000], marker='x', label=isotope)
    plt.title(key[:-1])
    plt.xlabel('M')
    plt.ylabel('air broadening')
    plt.legend()
    plt.show()
    break
    print('~~~~~~~~~~~~~~~~~~~~~~~~')
#%%
