#%%
import pandas as pd
import numpy as np
import os
from glob import iglob

rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/hitran_data/**/*' # Note the added asterisks
# This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-10:] == "/1_iso.csv" if "readme" not in f]
#%%
print(file_list)
#%%
db = {}

for f in file_list:
    i = f[76:-10]
    db[i] = pd.read_csv(f)
#%%
print(len(db))
del db['SO3']
#%%
for key, data in db.items():
    db[key] = data.sample(frac=1)
#%%
import matplotlib.pyplot as plt
#%%
for key, data in db.items():
    print(key)
#%%
molecules = []
error_2s = []
datapoints = []
for key, data in db.items():
    print(key)
    molecules.append(key)
    x = data['gamma_air-err']
    y = x.value_counts().sort_index()

    if 2 in y.index:
        print('percentage of data with error code 3 or above =')
        print((1-y.cumsum()[2]/x.count()).round(2))
        print('out of')
        print(x.count())
        print('datapoints')
        error_2s.append(1-y.cumsum()[2]/x.count())
        datapoints.append(x.count())
    else:
        print('percentage of data with error code 3 or above =')
        print(1-0.00)
        print('out of')
        print(x.count())
        print('datapoints')
        error_2s.append(1-0)
        datapoints.append(x.count())
    #print(100*x.value_counts()[2]/x.count())
#%%
ditfim = pd.DataFrame(np.array([error_2s, datapoints]).T, index=molecules)
import seaborn as sns

cm = sns.light_palette('green', as_cmap=True)

s = ditfim.style.background_gradient(cmap=cm, low=0, high=1, axis=0)
s
#%%
e2 = np.array(error_2s)
dat = np.array(datapoints)
good_data = e2*dat
print('percentage "good" data =')
print(sum(good_data)/sum(dat))

#%%
for key, data in db.items():
    print(key)
    print(data.columns[50:-16])
#%%
def broadening(m, T, ma, mp, b0):
    gamma = 1.7796e-5 * (m/(m-2)) * (1/np.sqrt(T)) * np.sqrt((ma+mp)/(ma*mp)) * b0**2
    return(gamma)
#%%
broadening(6, 298, 5, 10, 300)
#%%


for key, data in db.items():

    err_codes = data['gamma_air-err'].value_counts().sort_index()
    print(key)
    print(err_codes)
    data_by_vib_lev = {}

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for code in err_codes.index:
        data_level_x = data[data['gamma_air-err']==code]
        label = str(code)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)

    plt.legend()
    plt.show()
#%%
for key, data in db.items():

    vib_modes = []
    vib_levels = []
    print(key)
    for item in data.columns:
        if item[0] == 'v':
            if len(item)<5:
                vib_modes.append(item)

    for mode in vib_modes:
        vib_levels.append([mode, data[mode].value_counts()])

    data_by_vib_lev = {}
    
    for mode in vib_levels:
        levels = []
        for level in mode[1].index:
            levels.append(level)

        data_by_vib_lev[mode[0]] = levels

    print(data_by_vib_lev)
    break
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for mode, levels in data_by_vib_lev.items():
        for level in levels:
            data_level_x = data[data[mode]==level]
            label = mode+'='+str(level)
            plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.legend()
    plt.show()
#%%
for key, data in db.items():

    vib_modes = []
    vib_levels = []
    print(key)
    for item in data.columns:
        if item == 'ElecStateLabel':
            vib_modes.append(item)

    if len(vib_modes) == 0:
        continue

    for mode in vib_modes:
        vib_levels.append([mode, data[mode].value_counts()])

    if len(vib_levels[0][1]) == 1:
        continue

    data_by_vib_lev = {}
    
    for mode in vib_levels:
        levels = []
        for level in mode[1].index:
            levels.append(level)

        data_by_vib_lev[mode[0]] = levels

    print(data_by_vib_lev)
    
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for mode, levels in data_by_vib_lev.items():
        for level in levels:
            data_level_x = data[data[mode]==level]
            label = mode+'='+str(level)
            plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.legend()
    plt.show()
#%%

#%%

#%%
for key, data in db.items():

    vib_modes = []
    vib_levels = []
    print(key)
    for item in data.columns:
        if item[0] == 'K':
            vib_modes.append(item)


    if len(vib_modes) == 0:
        continue

    for mode in vib_modes:
        vib_levels.append([mode, data[mode].value_counts()])

    if len(vib_levels[0][1]) == 1:
        continue


    data_by_vib_lev = {}
    
    for mode in vib_levels:
        levels = []
        for level in mode[1].index:
            levels.append(level)

        data_by_vib_lev[mode[0]] = levels

    
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for mode, levels in data_by_vib_lev.items():
        for level in levels:
            data_level_x = data[data[mode]==level]
            label = mode+'='+str(level)
            plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    #plt.legend()
    plt.show()
#%%
for key, data in db.items():

    vib_modes = []
    vib_levels = []
    print(key)
    for item in data.columns:
        if item == 'v':
            vib_modes.append(item)


    if len(vib_modes) == 0:
        continue

    for mode in vib_modes:
        vib_levels.append([mode, data[mode].value_counts()])

    if len(vib_levels[0][1]) == 1:
        continue


    data_by_vib_lev = {}
    
    for mode in vib_levels:
        levels = []
        for level in mode[1].index:
            levels.append(level)

        data_by_vib_lev[mode[0]] = levels

    
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for mode, levels in data_by_vib_lev.items():
        for level in levels:
            data_level_x = data[data[mode]==level]
            label = mode+'='+str(level)
            plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.legend()
    plt.show()
#%%
