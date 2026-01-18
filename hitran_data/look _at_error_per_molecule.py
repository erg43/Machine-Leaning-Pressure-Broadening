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
# weight data by error code, currently error code = weighting
for molecule, data in db.items():
    # take weight as gamma-air-err
    weight = data['gamma_air-err']
    # Give helpful weightings
    # reweight 0 to tiny, because 0 gives /0 error
    
    weight = weight.replace(0, np.NaN)    # 0  ~~~  unreported or unavailable
    weight = weight.replace(1, np.NaN)    # 1  ~~~  Default or constant
    weight = weight.replace(2, np.NaN)    # 2  ~~~  Average or estimate
    weight = weight.replace(3, np.NaN)     # 3  ~~~  err >= 20 %              50
    weight = weight.replace(4, 20)     # 4  ~~~  20 >= err >= 10 %        15
    weight = weight.replace(5, 10)    # 5  ~~~  10 >= err >= 5 %         7.5
    weight = weight.replace(6, 5)    # 6  ~~~  5 >= err >= 2 %          3.5
    weight = weight.replace(7, 2)    # 7  ~~~  2 >= err >= 1 %          1.5
    weight = weight.replace(8, 1)    # 8  ~~~  err <= 1 %               0.5
    
    # reassign weight into dictionary
    db[molecule]['gamma_air-err'] = weight
#%%
molecules = []
error_2s = []
datapoints = []
molecules_error = []
mean_vals = []
error_plot = {}

for key, data in db.items():
    print()
    print()
    print(key)
    molecules.append(key)
    x = data['gamma_air-err']
    print()

    dat_err = 1-x.isna().sum()/len(x)
    print("The amount of data with an upper bound to its error is")
    print(dat_err)
    print('i.e. '+str(round(dat_err*100, 2))+"% of data is good")
    print()

    if dat_err == 1:
        molecules_error.append(key)
        mean_vals.append(np.mean(x))
        print('all data is good, mean error = '+str(round(np.mean(x)))+'%')
        error_plot[key] = [np.mean(x), np.mean(x), data['molecule_weight'][0]]
    else:
        if dat_err ==0:
            print("all data bad, mean error = 50%+")
            error_plot[key] = [np.nan, 50, data['molecule_weight'][0]]
        else:
            print('for the good data, mean error = '+str(round(np.mean(x)))+'%')
            print('assuming bad data has an error of 50%, total error ='+str(round(np.mean(x)*(1-x.isna().sum()/len(x)) + 50*x.isna().sum()/len(x)))+'%')
            error_plot[key] = [np.mean(x), np.mean(x)*(1-x.isna().sum()/len(x))+ 50*x.isna().sum()/len(x), data['molecule_weight'][0]]

    y = x.value_counts().sort_index()
    print(y)
    print()
    
    '''
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
    '''
    
print('molecules with accuracy defined data:')
print(molecules_error)
#%%
ploterr = pd.DataFrame.from_dict(error_plot, orient='index').sort_values(by=[1])
#%%
ploterr
#%%
fig1 = plt.figure(figsize=(30, 6))


plt.plot(ploterr.index, ploterr[0], 'x')
plt.plot(ploterr.index, ploterr[1], 'x')

plt.ylabel(r'Percentage error in Pressure Broadening, $\gamma_{Air}$ / $cm^{-1}atm^{-1}$')
plt.legend(loc='upper right')

plt.xlabel('Molecule')
plt.title(f'Accuracy of Pressure Broadening')

plt.legend()
plt.show()


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
