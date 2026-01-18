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
print(len(db))
del db['SO3']
#%%
for key, data in db.items():
    db[key] = data.sample(frac=1)
#%%
import matplotlib.pyplot as plt
#%%
molecules = []
error_2s = []
datapoints = []
for key, data in db.items():
    #print(key)
    molecules.append(key)
    x = data['gamma_air-err']
    y = x.value_counts().sort_index()

    if 2 in y.index:
        #print('percentage of data with error code 3 or above =')
        #print((1-y.cumsum()[2]/x.count()).round(2))
        #print('out of')
        #print(x.count())
        #print('datapoints')
        error_2s.append(1-y.cumsum()[2]/x.count())
        datapoints.append(x.count())
    else:
        #print('percentage of data with error code 3 or above =')
        #print(1-0.00)
        #print('out of')
        #print(x.count())
        #print('datapoints')
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
def broadening(m, T, ma, mp, b0):
    gamma = 1.7796e-5 * (m/(m-2)) * (1/np.sqrt(T)) * np.sqrt((ma+mp)/(ma*mp)) * b0**2
    return(gamma)
#%%
T = 298
#%%
molecules = {}

for key, data in db.items():

    if 'B0a' in data.columns:
        data = data[['gamma_air', 'J', 'Jpp', 'molecule_weight', 'gamma_air-err', 'm', 'findair', 'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight']]
        molecules[key] = data
        broadness_jeanna = broadening(data['m'][0], T, data['molecule_weight'][0], data['air_weight'][0], data['findair'][0])
        molecules[key]['broadness_jeanna'] = broadness_jeanna
    '''if 'findair' in data.columns:
        if 'K' in data.columns:
            data['Kval'] = data['K']
            molecules[key] = data
            
            broadness_jeanna = broadening(data['m'][0], T, data['molecule_weight'][0], data['air_weight'][0], data['findair'][0])
            molecules[key]['broadness_jeanna'] = broadness_jeanna

        elif 'Ka' in data.columns:
            data['Kval'] = data['Ka']
            molecules[key] = data
            
            broadness_jeanna = broadening(data['m'][0], T, data['molecule_weight'][0], data['air_weight'][0], data['findair'][0])
            molecules[key]['broadness_jeanna'] = broadness_jeanna'''
#%%
print(molecules.keys())
#%%
import random
molecule_list = {}

count = 0
for molecule in molecules:
    if count < 10:
        print(molecule)
        test_data = molecules[molecule]
        train_data = set(molecules) - {molecule}
        #print(test_data['nu'])
        #for k in test_data:
        #    print([k])

        test_data = {molecule: molecules[molecule] for k in [test_data]}
        train_data = {k: molecules[k] for k in train_data}

        data_train = pd.concat([train_data[k] for k in train_data])
        data_test = pd.concat([test_data[k] for k in test_data])

        molecule_list[molecule] = [data_train, data_test]

        count +=1
#%%
count = 0
for molecule in molecules:
    if count < 10:
        count +=1
        continue
    elif count < 20:
        print(molecule)
        test_data = molecules[molecule]
        train_data = set(molecules) - {molecule}
        #print(test_data['nu'])
        #for k in test_data:
        #    print([k])

        test_data = {molecule: molecules[molecule] for k in [test_data]}
        train_data = {k: molecules[k] for k in train_data}

        data_train = pd.concat([train_data[k] for k in train_data])
        data_test = pd.concat([test_data[k] for k in test_data])

        molecule_list[molecule] = [data_train, data_test]

        count +=1
#%%
count = 0
for molecule in molecules:
    if count < 20:
        count +=1
        continue
    elif count < 30:
        print(molecule)
        test_data = molecules[molecule]
        train_data = set(molecules) - {molecule}
        #print(test_data['nu'])
        #for k in test_data:
        #    print([k])

        test_data = {molecule: molecules[molecule] for k in [test_data]}
        train_data = {k: molecules[k] for k in train_data}

        data_train = pd.concat([train_data[k] for k in train_data])
        data_test = pd.concat([test_data[k] for k in test_data])

        molecule_list[molecule] = [data_train, data_test]

        count +=1
#%%
count = 0
for molecule in molecules:
    if count < 30:
        count +=1
        continue
    elif count < 40:
        print(molecule)
        test_data = molecules[molecule]
        train_data = set(molecules) - {molecule}
        #print(test_data['nu'])
        #for k in test_data:
        #    print([k])

        test_data = {molecule: molecules[molecule] for k in [test_data]}
        train_data = {k: molecules[k] for k in train_data}

        data_train = pd.concat([train_data[k] for k in train_data])
        data_test = pd.concat([test_data[k] for k in test_data])

        molecule_list[molecule] = [data_train, data_test]

        count +=1
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#%%
for molecule in molecule_list:
    print(molecule)
    data = molecule_list[molecule]

    data_train = data[0]#[['nu', 'sw', 'a', 'gamma_air', 'J', 'Jpp', 'molecule_weight', 'air_weight', 'gamma_air-err', 'm', 'findair', 'broadness_jeanna', 'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c']]
    data_test = data[1]#[['nu', 'sw', 'a', 'gamma_air', 'J', 'Jpp', 'molecule_weight', 'air_weight', 'gamma_air-err', 'm', 'findair', 'broadness_jeanna', 'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c']]


    points = 639772//len(data_test)
    print(points)

    data_test = pd.concat([data_test]*points)
    print(len(data_test))
    molecule_list[molecule][0] = data_train
    molecule_list[molecule][1] = data_test
#%%
for molecule in molecule_list:
    data_test = molecule_list[molecule][1]

    weight_test = data_test['gamma_air-err']
    #print(weight_test)
    weight_test=weight_test.replace(0,0.0001)
    molecule_list[molecule][1]['gamma_air-err'] = weight_test

#%%
from sklearn.ensemble import HistGradientBoostingRegressor
#%%
for molecule in molecules:
    figure(figsize=((15, 7)), dpi=500)
    print(molecule)
    fig1 = plt.figure(1)
    #frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    #for code in err_codes.index:
    #data_level_x = data_test[data_test['gamma_air-err']==code]
    #label = str(code)
    #plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
    plt.plot(nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
    plt.xlabel('J, rotational quantum number')
    plt.ylabel('Gamma, line broadening /cm-1·atm-1')
    plt.legend()
    #frame2=fig1.add_axes((1, .1, 2.5, 1))
    #plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.savefig('diatomics_'+molecule+'_prediction.pdf')

    #broadness_jeanna = broadening(data_test['m'][0], T, data_test['molecule_weight'][0], data_test['air_weight'][0], data_test['findair'][0])
    plt.axhline(y=broadness_jeanna, linestyle='-')




    plt.show()


    err_codes = data_test['gamma_air-err'].value_counts().sort_index()
    print(molecule)
    print(err_codes)
    data_by_vib_lev = {}

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for code in err_codes.index:
        data_level_x = data_test[data_test['gamma_air-err']==code]
        label = str(code)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)

    plt.legend()
    plt.show()
#%%
total = 0

for molecule in molecule_list:
    data = molecule_list[molecule]
    data_train = data[0]
    data_test = data[1]

    broadness_jeanna = broadening(data_test['m'].values[0], T, data_test['molecule_weight'].values[0], data_test['air_weight'].values[0], data_test['findair'].values[0])
    print(broadness_jeanna)

    print('train')
    print(data_train.isnull().values.any())
    print('test')
    print(data_test.isnull().values.any())

    data_train = data_train.dropna()
    data_test = data_test.dropna()
    print('train')
    print(data_train.isnull().values.any())

    data_test = data_test.sample(frac=1)
    data_train = data_train.sample(frac=1)
    X_train = data_train.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_train = data_train['gamma_air']
    weight_train = data_train['gamma_air-err']
    X_test = data_test.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_test = data_test['gamma_air']
    weight_test = data_test['gamma_air-err']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pipe = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
    pipe.fit(X_train, y_train, histgradientboostingregressor__sample_weight=weight_train)#**{GradientBoostingRegressor__sample_weight: weight_train})

    y_pred = pipe.predict(X_test)

    print('score = '+str(pipe.score(X_test, y_test, weight_test)))
    total += pipe.score(X_test, y_test)
    y_test_plot = y_test.to_numpy()

    #print(data_dun.columns)

    nu_plot = X_test['J'].to_numpy()


    yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

    from matplotlib.pyplot import figure
    
    #err_codes = data['gamma_air-err'].value_counts().sort_index()


    figure(figsize=((15, 7)), dpi=500)
    print(molecule)
    fig1 = plt.figure(1)
    #frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    #for code in err_codes.index:
    #data_level_x = data_test[data_test['gamma_air-err']==code]
    #label = str(code)
    #plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
    plt.plot(nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
    plt.xlabel('J, rotational quantum number')
    plt.ylabel('Gamma, line broadening /cm-1·atm-1')
    plt.legend()
    #frame2=fig1.add_axes((1, .1, 2.5, 1))
    #plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.savefig('diatomics_'+molecule+'_prediction.pdf')

    #broadness_jeanna = broadening(data_test['m'][0], T, data_test['molecule_weight'][0], data_test['air_weight'][0], data_test['findair'][0])
    plt.axhline(y=broadness_jeanna, linestyle='-')




    plt.show()


    err_codes = data_test['gamma_air-err'].value_counts().sort_index()
    print(molecule)
    print(err_codes)
    data_by_vib_lev = {}

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for code in err_codes.index:
        data_level_x = data_test[data_test['gamma_air-err']==code]
        label = str(code)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)

    plt.legend()
    plt.show()
#%%
total = 0

for molecule in molecule_list:
    data = molecule_list[molecule]
    data_train = data[0]
    data_test = data[1]

    broadness_jeanna = broadening(data_test['m'][0].values[0], T, data_test['molecule_weight'][0].values[0], data_test['air_weight'][0].values[0], data_test['findair'][0].values[0])
    print(broadness_jeanna)

    print('train')
    print(data_train.isnull().values.any())
    print('test')
    print(data_test.isnull().values.any())

    data_train = data_train.dropna()
    data_test = data_test.dropna()
    print('train')
    print(data_train.isnull().values.any())

    data_test = data_test.sample(frac=1)
    data_train = data_train.sample(frac=1)
    X_train = data_train.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_train = data_train['gamma_air']
    weight_train = data_train['gamma_air-err']
    X_test = data_test.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_test = data_test['gamma_air']
    weight_test = data_test['gamma_air-err']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    pipe.fit(X_train, y_train, gradientboostingregressor__sample_weight=weight_train)#**{GradientBoostingRegressor__sample_weight: weight_train})

    y_pred = pipe.predict(X_test)

    print('score = '+str(pipe.score(X_test, y_test, weight_test)))
    total += pipe.score(X_test, y_test)
    y_test_plot = y_test.to_numpy()

    #print(data_dun.columns)

    nu_plot = X_test['J'].to_numpy()


    yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

    from matplotlib.pyplot import figure
    
    #err_codes = data['gamma_air-err'].value_counts().sort_index()


    figure(figsize=((15, 7)), dpi=500)
    print(molecule)
    fig1 = plt.figure(1)
    #frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    #for code in err_codes.index:
    #data_level_x = data_test[data_test['gamma_air-err']==code]
    #label = str(code)
    #plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
    plt.plot(nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
    plt.xlabel('J, rotational quantum number')
    plt.ylabel('Gamma, line broadening /cm-1·atm-1')
    plt.legend()
    #frame2=fig1.add_axes((1, .1, 2.5, 1))
    #plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.savefig('diatomics_'+molecule+'_prediction.pdf')

    #broadness_jeanna = broadening(data_test['m'][0], T, data_test['molecule_weight'][0], data_test['air_weight'][0], data_test['findair'][0])
    plt.axhline(y=broadness_jeanna, linestyle='-')




    plt.show()


    err_codes = data_test['gamma_air-err'].value_counts().sort_index()
    print(molecule)
    print(err_codes)
    data_by_vib_lev = {}

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for code in err_codes.index:
        data_level_x = data_test[data_test['gamma_air-err']==code]
        label = str(code)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)

    plt.legend()
    plt.show()
#%%
total = 0
count =0
for molecule in molecule_list:
    if count <7:
        count+=1
        continue
    data = molecule_list[molecule]
    data_train = data[0]
    data_test = data[1]

    broadness_jeanna = broadening(data_test['m'].values[0], T, data_test['molecule_weight'].values[0], data_test['air_weight'].values[0], data_test['findair'].values[0])
    print(broadness_jeanna)

    print('train')
    print(data_train.isnull().values.any())
    print('test')
    print(data_test.isnull().values.any())

    data_train = data_train.dropna()
    data_test = data_test.dropna()
    print('train')
    print(data_train.isnull().values.any())

    data_test = data_test.sample(frac=1)
    data_train = data_train.sample(frac=1)
    X_train = data_train.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_train = data_train['gamma_air']
    weight_train = data_train['gamma_air-err']
    X_test = data_test.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_test = data_test['gamma_air']
    weight_test = data_test['gamma_air-err']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    pipe.fit(X_train, y_train, gradientboostingregressor__sample_weight=weight_train)#**{GradientBoostingRegressor__sample_weight: weight_train})

    y_pred = pipe.predict(X_test)

    print('score = '+str(pipe.score(X_test, y_test, weight_test)))
    total += pipe.score(X_test, y_test)
    y_test_plot = y_test.to_numpy()

    #print(data_dun.columns)

    nu_plot = X_test['J'].to_numpy()


    yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

    from matplotlib.pyplot import figure
    
    #err_codes = data['gamma_air-err'].value_counts().sort_index()


    figure(figsize=((15, 7)), dpi=500)
    print(molecule)
    fig1 = plt.figure(1)
    #frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    #for code in err_codes.index:
    #data_level_x = data_test[data_test['gamma_air-err']==code]
    #label = str(code)
    #plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
    plt.plot(nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
    plt.xlabel('J, rotational quantum number')
    plt.ylabel('Gamma, line broadening /cm-1·atm-1')
    plt.legend()
    #frame2=fig1.add_axes((1, .1, 2.5, 1))
    #plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.savefig('diatomics_'+molecule+'_prediction.pdf')

    #broadness_jeanna = broadening(data_test['m'][0], T, data_test['molecule_weight'][0], data_test['air_weight'][0], data_test['findair'][0])
    plt.axhline(y=broadness_jeanna, linestyle='-')




    plt.show()


    err_codes = data_test['gamma_air-err'].value_counts().sort_index()
    print(molecule)
    print(err_codes)
    data_by_vib_lev = {}

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for code in err_codes.index:
        data_level_x = data_test[data_test['gamma_air-err']==code]
        label = str(code)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)

    plt.legend()
    plt.show()
#%%
total = 0

for molecule in molecule_list:
    data = molecule_list[molecule]
    data_train = data[0]
    data_test = data[1]

    broadness_jeanna = broadening(data_test['m'].values[0], T, data_test['molecule_weight'].values[0], data_test['air_weight'].values[0], data_test['findair'].values[0])
    print(broadness_jeanna)

    print('train')
    print(data_train.isnull().values.any())
    print('test')
    print(data_test.isnull().values.any())

    data_train = data_train.dropna()
    data_test = data_test.dropna()
    print('train')
    print(data_train.isnull().values.any())

    data_test = data_test.sample(frac=1)
    data_train = data_train.sample(frac=1)
    X_train = data_train.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_train = data_train['gamma_air']
    weight_train = data_train['gamma_air-err']
    X_test = data_test.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_test = data_test['gamma_air']
    weight_test = data_test['gamma_air-err']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    pipe.fit(X_train, y_train, gradientboostingregressor__sample_weight=weight_train)#**{GradientBoostingRegressor__sample_weight: weight_train})

    y_pred = pipe.predict(X_test)

    print('score = '+str(pipe.score(X_test, y_test, weight_test)))
    total += pipe.score(X_test, y_test)
    y_test_plot = y_test.to_numpy()

    #print(data_dun.columns)

    nu_plot = X_test['J'].to_numpy()


    yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

    from matplotlib.pyplot import figure
    
    #err_codes = data['gamma_air-err'].value_counts().sort_index()


    figure(figsize=((15, 7)), dpi=500)
    print(molecule)
    fig1 = plt.figure(1)
    #frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    #for code in err_codes.index:
    #data_level_x = data_test[data_test['gamma_air-err']==code]
    #label = str(code)
    #plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
    plt.plot(nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
    plt.xlabel('J, rotational quantum number')
    plt.ylabel('Gamma, line broadening /cm-1·atm-1')
    plt.legend()
    #frame2=fig1.add_axes((1, .1, 2.5, 1))
    #plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.savefig('diatomics_'+molecule+'_prediction.pdf')

    #broadness_jeanna = broadening(data_test['m'][0], T, data_test['molecule_weight'][0], data_test['air_weight'][0], data_test['findair'][0])
    plt.axhline(y=broadness_jeanna, linestyle='-')




    plt.show()


    err_codes = data_test['gamma_air-err'].value_counts().sort_index()
    print(molecule)
    print(err_codes)
    data_by_vib_lev = {}

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for code in err_codes.index:
        data_level_x = data_test[data_test['gamma_air-err']==code]
        label = str(code)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)

    plt.legend()
    plt.show()
#%% md
# # ALL MOLECULES, EVEN LINEAR
#%%
total = 0

for molecule in molecule_list:
    data = molecule_list[molecule]
    data_train = data[0]
    data_test = data[1]

    broadness_jeanna = broadening(data_test['m'].values[0], T, data_test['molecule_weight'].values[0], data_test['air_weight'].values[0], data_test['findair'].values[0])
    print(broadness_jeanna)

    print('train')
    print(data_train.isnull().values.any())
    print('test')
    print(data_test.isnull().values.any())

    data_train = data_train.dropna()
    data_test = data_test.dropna()
    print('train')
    print(data_train.isnull().values.any())

    data_test = data_test.sample(frac=1)
    data_train = data_train.sample(frac=1)
    X_train = data_train.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_train = data_train['gamma_air']
    weight_train = data_train['gamma_air-err']
    X_test = data_test.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_test = data_test['gamma_air']
    weight_test = data_test['gamma_air-err']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    pipe.fit(X_train, y_train, gradientboostingregressor__sample_weight=weight_train)#**{GradientBoostingRegressor__sample_weight: weight_train})

    y_pred = pipe.predict(X_test)

    print('score = '+str(pipe.score(X_test, y_test, weight_test)))
    total += pipe.score(X_test, y_test)
    y_test_plot = y_test.to_numpy()

    #print(data_dun.columns)

    nu_plot = X_test['J'].to_numpy()


    yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

    from matplotlib.pyplot import figure
    
    #err_codes = data['gamma_air-err'].value_counts().sort_index()


    '''
    print(molecule)
    fig1 = plt.figure(1)
    #frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    #for code in err_codes.index:
    #data_level_x = data_test[data_test['gamma_air-err']==code]
    #label = str(code)
    #plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
    plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
    plt.plot(nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
    plt.xlabel('J, rotational quantum number')
    plt.ylabel(r'$\gamma_{air}$, line broadening /cm^{-1}·atm^{-1}')
    plt.legend()
    #frame2=fig1.add_axes((1, .1, 2.5, 1))
    #plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.savefig('diatomics_'+molecule+'_prediction.pdf')

    #broadness_jeanna = broadening(data_test['m'][0], T, data_test['molecule_weight'][0], data_test['air_weight'][0], data_test['findair'][0])
    '''




    Air = 'Air'
    import math

    figure(figsize=((15, 7)), dpi=500)
    fig1 = plt.figure(1)
    print(molecule)

    frame1=fig1.add_axes((1, 0, 2.5, 1))
    plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
    plt.plot( nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
    plt.axhline(y=broadness_jeanna, linestyle='-', label='Pressure broadening predicted by the Buldyreva formalism')
    plt.ylabel(r'Pressure Broadening, $\gamma_{air}$ / $cm^{-1}atm^{-1}$')
    plt.legend(loc='upper right')
    new_list = range(math.floor(min(nu_plot[:200])), math.ceil(max(nu_plot[:200]))+1, 2)
    plt.xticks(new_list)
    plt.xlabel('J, rotational quantum number')
    plt.title(f'Predicted $\gamma_{{{Air}}}$ for lines in {molecule}')

    plt.show()


    err_codes = data_test['gamma_air-err'].value_counts().sort_index()
    print(molecule)
    print(err_codes)
    data_by_vib_lev = {}

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    for code in err_codes.index:
        data_level_x = data_test[data_test['gamma_air-err']==code]
        label = str(code)
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)

    plt.legend()
    plt.show()
#%%
total = 0

for molecule in molecule_list:
    if molecule in ['CS', 'H2O', 'CO']:
        data = molecule_list[molecule]
        data_train = data[0]
        data_test = data[1]

        broadness_jeanna = broadening(data_test['m'].values[0], T, data_test['molecule_weight'].values[0], data_test['air_weight'].values[0], data_test['findair'].values[0])
        print(broadness_jeanna)

        print('train')
        print(data_train.isnull().values.any())
        print('test')
        print(data_test.isnull().values.any())

        data_train = data_train.dropna()
        data_test = data_test.dropna()
        print('train')
        print(data_train.isnull().values.any())

        data_test = data_test.sample(frac=1)
        data_train = data_train.sample(frac=1)
        X_train = data_train.drop(['gamma_air', 'gamma_air-err'], axis=1)
        y_train = data_train['gamma_air']
        weight_train = data_train['gamma_air-err']
        X_test = data_test.drop(['gamma_air', 'gamma_air-err'], axis=1)
        y_test = data_test['gamma_air']
        weight_test = data_test['gamma_air-err']

        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
        pipe.fit(X_train, y_train, gradientboostingregressor__sample_weight=weight_train)#**{GradientBoostingRegressor__sample_weight: weight_train})

        y_pred = pipe.predict(X_test)

        print('score = '+str(pipe.score(X_test, y_test, weight_test)))
        total += pipe.score(X_test, y_test)
        y_test_plot = y_test.to_numpy()

        #print(data_dun.columns)

        nu_plot = X_test['J'].to_numpy()


        yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

        from matplotlib.pyplot import figure
        
        #err_codes = data['gamma_air-err'].value_counts().sort_index()


        '''
        print(molecule)
        fig1 = plt.figure(1)
        #frame1=fig1.add_axes((1, 1.1, 2.5, 1))
        #for code in err_codes.index:
        #data_level_x = data_test[data_test['gamma_air-err']==code]
        #label = str(code)
        #plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)
        plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
        plt.plot(nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
        plt.xlabel('J, rotational quantum number')
        plt.ylabel(r'$\gamma_{air}$, line broadening /cm^{-1}·atm^{-1}')
        plt.legend()
        #frame2=fig1.add_axes((1, .1, 2.5, 1))
        #plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
        #plt.savefig('diatomics_'+molecule+'_prediction.pdf')

        #broadness_jeanna = broadening(data_test['m'][0], T, data_test['molecule_weight'][0], data_test['air_weight'][0], data_test['findair'][0])
        '''




        Air = 'Air'
        import math


        fig1 = plt.figure(1)
        print(molecule)

        frame1=fig1.add_axes((1, 1.1, 2.5, 1))
        plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
        plt.plot( nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
        plt.axhline(y=broadness_jeanna, linestyle='-', label='Pressure broadening predicted by the Buldyreva formalism')
        plt.ylabel(r'Pressure Broadening, $\gamma_{air}$ / $cm^{-1}atm^{-1}$')
        plt.legend(loc='upper right')
        new_list = range(math.floor(min(nu_plot[:200])), math.ceil(max(nu_plot[:200]))+1, 2)
        plt.xticks(new_list)
        plt.xlabel('J, rotational quantum number')
        plt.title(f'Predicted $\gamma_{{{Air}}}$ for lines in {molecule}')

        plt.show()


        err_codes = data_test['gamma_air-err'].value_counts().sort_index()
        print(molecule)
        print(err_codes)
        data_by_vib_lev = {}

        fig1 = plt.figure(1)
        frame1=fig1.add_axes((1, 1.1, 2.5, 1))
        for code in err_codes.index:
            data_level_x = data_test[data_test['gamma_air-err']==code]
            label = str(code)
            plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label=label)

        plt.legend()
        plt.show()
#%%
print(model)
#%%
pipe[1].get_params()
#%%
pipe[1].estimators_
#%%
pipe[1].feature_importances_
#%%
pipe[1].get_params()
#%%
pipe[1].feature_importances_
#%%
pipe[1].estimators_
#%%
