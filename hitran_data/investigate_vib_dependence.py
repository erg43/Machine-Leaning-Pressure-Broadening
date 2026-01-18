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
db.pop('HONO_prediction')
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
    print(data.columns)
#%%
for key, data in db.items():

    err_codes = data['gamma_air-err'].value_counts().sort_index()
    #print(key)
    #print(err_codes)
    data_by_vib_lev = {}
    if key != 'C2H6':
        continue


    Air = 'Air'
    import math


    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for code in err_codes.index:
        data_level_x = data[data['gamma_air-err']==code]
        label = 'error code = '+str(code)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)


        plt.ylabel(r'Pressure Broadening, $\gamma_{Air}$ / $cm^{-1}atm^{-1}$')
        plt.legend(loc='upper right')

        plt.xlabel('J, rotational quantum number')
        plt.title(f'Accuracy of each $\gamma_{{{Air}}}$ value in {key}')

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
for key, data in db.items():
    print(key)
    if 'K' in data.columns:
        print("K")
    if 'Kc' in data.columns:
        print("Kc")
    if 'Ka' in data.columns:
        print("Ka")

#%%
for key, data in db.items():

    vib_modes = []
    vib_levels = []
    print(key)
    for item in data.columns:
        if item == 'K':
            vib_modes.append(item)
        #elif item == 'Ka':
        #    vib_modes.append(item)

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

    data_by_vib_lev['K'].sort()


    Air = 'Air'
    import math




    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for mode, levels in data_by_vib_lev.items():
        for level in levels:
            data_level_x = data[data[mode]==level]
            label = mode+'='+str(level)
            plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)


    plt.ylabel(r'Pressure Broadening, $\gamma_{Air}$ / $cm^{-1}atm^{-1}$')
    plt.legend(loc='upper right')
    plt.xlim(-1, 44)
    plt.xlabel('J, rotational quantum number')
    plt.title(f'Effect of K on $\gamma_{{{Air}}}$ values in {key}')

    plt.legend()
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

    Air='Air'
    import math
    data_by_vib_lev['v'].sort()

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for mode, levels in data_by_vib_lev.items():
        for level in levels:
            data_level_x = data[data[mode]==level]
            label = mode+'='+str(level)
            plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.ylabel(r'Pressure Broadening, $\gamma_{Air}$ / $cm^{-1}atm^{-1}$')
    plt.legend(ncol=2, loc='upper right')

    plt.xlabel('J, rotational quantum number')
    plt.title(f'Effect of vibrational state on $\gamma_{{{Air}}}$ values in {key}')

    plt.show()


#%%
ethane = db['C2H6']
#%%
ethane.columns
#%%
ethane = ethane.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
#%%
ethane.columns
#%%
ethane = ethane.drop(['H2O_dipole', 'm', 'd', 'dair', 'findair', 'molecule_dipole', 'polar',
       'B0a', 'B0b', 'B0c'], axis=1)
#%%
ethane.columns
#%%
ethane = ethane.drop(['molecule_weight', 'self_weight', 'air_weight',
       'He_weight', 'H2_weight', 'CO2_weight', 'H2O_weight',
       'molecule_quadrupole_z', 'molecule_quadrupole_x',
       'molecule_quadrupole_y', 'self_quadrupole_z', 'self_quadrupole_x',
       'self_quadrupole_y', 'air_quadrupole_z', 'air_quadrupole_x',
       'H2_quadrupole_z', 'H2_quadrupole_x', 'CO2_quadrupole_z',
       'CO2_quadrupole_x'], axis=1)
#%%
ethane
#%%
ethane = ethane.drop(['global_iso_id', 'molec_id', 'local_iso_id', 'nu', 'sw', 'a', 'elower',
       'gp', 'gpp'], axis=1)
#%%
ethane.columns
#%%
ethane = ethane.drop(['n_air', 'gamma_self', 'n_self', 'gamma_H2', 'n_H2',
       'gamma_He', 'n_He', 'gamma_CO2', 'n_CO2', 'gamma_H2O', 'n_H2O',
       'nu-err', 'sw-err', 'n_air-err', 'gamma_self-err',
       'n_self-err', 'gamma_H2-err', 'n_H2-err', 'gamma_He-err', 'n_He-err',
       'gamma_CO2-err', 'n_CO2-err', 'gamma_H2O-err', 'n_H2O-err', 'nu-ref',
       'sw-ref', 'n_air-ref', 'gamma_self-ref', 'n_self-ref',
       'gamma_H2-ref', 'n_H2-ref', 'gamma_He-ref', 'n_He-ref', 'gamma_CO2-ref',
       'n_CO2-ref', 'gamma_H2O-ref', 'n_H2O-ref'], axis=1)
#%%
ethane
#%%
data = ethane

vib_modes = []
vib_levels = []

for item in data.columns:
    if item[0] == 'v':
        if not item[-1] == 'p':
            if not item == 'v4':
                vib_modes.append(item)
                ''' if not item == 'v4pp':
                if not item == 'v3pp':
                    if not item == 'v2pp':
                        if not item == 'v1pp':
                            vib_modes.append(item)'''



for mode in vib_modes:
    vib_levels.append([mode, data[mode].value_counts()])


data_by_vib_lev = {}

for mode in vib_levels:
    levels = []
    for level in mode[1].index:
        levels.append(level)

    data_by_vib_lev[mode[0]] = levels

Air='Air'
import math
#data_by_vib_lev['v1'].sort()

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
for mode, levels in data_by_vib_lev.items():
    for level in levels:
        data_level_x = data[data[mode]==level]
        label = mode+'='+str(level)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
plt.ylabel(r'Pressure Broadening, $\gamma_{Air}$ / $cm^{-1}atm^{-1}$')
plt.legend(ncol=2, loc='upper right')

plt.xlabel('J, rotational quantum number')
plt.title(f'Effect of vibrational state on $\gamma_{{{Air}}}$ values in {key}')

plt.show()


#%%
ethane.columns
#%%
data = ethane

key = 'ethane'


vib_modes = []
vib_levels = []

for item in data.columns:
    if item == 'l':
        vib_modes.append(item)



for mode in vib_modes:
    vib_levels.append([mode, data[mode].value_counts()])
    
    vib_levels[0][1] = vib_levels[0][1].drop(labels=0)


data_by_vib_lev = {}

for mode in vib_levels:
    levels = []
    for level in mode[1].index:
        levels.append(level)

    data_by_vib_lev[mode[0]] = levels

Air='Air'
import math
#data_by_vib_lev['v1'].sort()

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
for mode, levels in data_by_vib_lev.items():
    for level in levels:
        data_level_x = data[data[mode]==level]
        label = mode+'='+str(level)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
plt.ylabel(r'Pressure Broadening, $\gamma_{Air}$ / $cm^{-1}atm^{-1}$')
plt.legend(ncol=2, loc='upper right')

plt.xlabel('J, rotational quantum number')
plt.title(f'Effect of vibrational state on $\gamma_{{{Air}}}$ values in {key}')

plt.show()


#%%
ethane
#%%
data = ethane

key = 'ethane'


vib_modes = []
vib_levels = []

for item in data.columns:
    if item[0] == 'r':
        if not item[-1] == 'p':
            vib_modes.append(item)



for mode in vib_modes:
    vib_levels.append([mode, data[mode].value_counts()])
    
    #vib_levels[0][1] = vib_levels[0][1].drop(labels=[2, 3, 4, 8, 12, 13, 16, 20, 23, 30, 1, 29, 27,25, 28])
    #vib_levels[0][1] = vib_levels[0][1].iloc[[24, 25, 26, 27]]


data_by_vib_lev = {}

for mode in vib_levels:
    levels = []
    for level in mode[1].index:
        levels.append(level)

    data_by_vib_lev[mode[0]] = levels

Air='Air'
import math
#data_by_vib_lev['v1'].sort()

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
for mode, levels in data_by_vib_lev.items():
    for level in levels:
        data_level_x = data[data[mode]==level]
        label = mode+'='+str(level)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
plt.ylabel(r'Pressure Broadening, $\gamma_{Air}$ / $cm^{-1}atm^{-1}$')
plt.legend(ncol=2, loc='upper right')

plt.xlabel('J, rotational quantum number')
plt.title(f'Effect of vibrational state on $\gamma_{{{Air}}}$ values in {key}')

plt.show()


#%%
