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
print(file_list)
for f in file_list:
    i = f[76:-10]
    print(i)
    x = pd.read_csv(f)
    if i=="SO3":
        continue
    x = x[['nu', 'sw', 'a', 'gamma_air', 'J', 'Jpp', 'molecule_weight']]
    x=x.dropna(axis=0)
    db[i] = x
#%%
print(len(db))

#%%
for key, data in db.items():
    db[key] = data.sample(frac=1)
#%%
for key, data in db.items():
    if len(data) > 50000:
        db[key] = data.iloc[:50000, :]


#%%
import matplotlib.pyplot as plt
#%%
print(db["H2"].columns)
#%%
import copy

molecules = copy.deepcopy(db)
molecule_list = copy.deepcopy(db)
#%%
import random

for molecule in molecules:
    print(molecule, len(molecules[molecule]))

    test_data = {molecule}
    train_data = set(molecules) - {molecule}
    
    test_data = {k: molecules[k] for k in test_data}
    train_data = {k: molecules[k] for k in train_data}

    data_train = pd.concat([train_data[k] for k in train_data])
    data_test = pd.concat([test_data[k] for k in test_data])

    molecule_list[molecule] = [data_train, data_test]
    print("done")
#%%

#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#%%
print(len(molecule_list))
#%%
for molecule in molecule_list:
    print(molecule)
    data = molecule_list[molecule]
    data_train = data[0]
    data_test = data[1]

    points = 50000//len(data_test)
    print(points)

    data_test = pd.concat([data_test]*points)
    print(len(data_test))
    molecule_list[molecule][0] = data_train
    molecule_list[molecule][1] = data_test
#%%
total = 0

for molecule in molecule_list:
    if molecule in ['C2H6', 'CH3Br', 'CO']:
        data = molecule_list[molecule]
        data_train = data[0]
        data_test = data[1]

        data_test = data_test.sample(frac=1)
        data_train = data_train.sample(frac=1)
        X_train = data_train.drop(['gamma_air'], axis=1)
        y_train = data_train['gamma_air']
        X_test = data_test.drop(['gamma_air'], axis=1)
        y_test = data_test['gamma_air']

        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        print('score = '+str(pipe.score(X_test, y_test)))
        total += pipe.score(X_test, y_test)
        y_test_plot = y_test.to_numpy()

        #print(data_dun.columns)

        nu_plot = X_test['J'].to_numpy()


        yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

        Air = 'Air'
        import math

        fig1 = plt.figure(1)
        print(molecule)
        frame1=fig1.add_axes((1, 0, 2.5, 1))
        plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
        plt.plot(nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
        plt.ylabel(r'Pressure Broadening, $\gamma_{Air}$ / $cm^{-1}atm^{-1}$')
        plt.legend(loc='upper right')
        new_list = range(math.floor(min(nu_plot[:200])), math.ceil(max(nu_plot[:200]))+1, 2)
        plt.xticks(new_list)
        plt.xlabel('J, rotational quantum number')
        plt.title(f'Predicted $\gamma_{{{Air}}}$ for lines in {molecule}')

        plt.show()
#%%
Air = 'Air'
import math

fig1 = plt.figure(1)
print(molecule)
frame1=fig1.add_axes((1, 0, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', label="Pressure broadening given by HITRAN")
plt.plot( nu_plot[:200], y_pred[:200], 'bx', label="Predicted pressure broadening")
plt.ylabel(r'Pressure Broadening, $\gamma_{air}$ / $cm^{-1}atm^{-1}$')
plt.legend(loc='upper right')
new_list = range(math.floor(min(nu_plot[:200])), math.ceil(max(nu_plot[:200]))+1, 2)
plt.xticks(new_list)
plt.xlabel('J, rotational quantum number')
plt.title(f'Predicted $\gamma_{{{Air}}}$ for lines in {molecule}')

plt.show()
#%%
molecules[['H2O', 'CH3F']]
#%%
total = 0
dict_filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
data_train = dict_filter(molecules, ('H2O', 'CH3F', 'COCl2', 'C2N2', 'H2S', 'C4H2', 'HOBr', 'C2H6', 'CH3I', 'H2', 'CH3Br', 'HO2', 'HNO3', 'NO', 'HCN', 'CH3Cl', 'O2', 'SO2', 'OH', 'NH3', 'CH3CN', 'CO2', 'HBr'))
data_test = dict_filter(molecules, ('CO', 'CS', 'CH3OH', 'O3', 'HCOOH', 'ClO', 'PH3', 'HOCl', 'OCS', 'CH4', 'C2H4', 'SO', 'H2O2', 'COF2', 'N2O', 'H2CO', 'C2H2', 'HCl', 'HC3N', 'NO2', 'HF', 'CS2', 'HI', 'N2'))

data_test = pd.concat([test_data[k] for k in test_data])
data_train = pd.concat([train_data[k] for k in train_data])

data_test = data_test.sample(frac=1)
data_train = data_train.sample(frac=1)
X_train = data_train.drop(['gamma_air'], axis=1)
y_train = data_train['gamma_air']
X_test = data_test.drop(['gamma_air'], axis=1)
y_test = data_test['gamma_air']

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('score = '+str(pipe.score(X_test, y_test)))
total += pipe.score(X_test, y_test)
y_test_plot = y_test.to_numpy()

#print(data_dun.columns)

nu_plot = X_test['J'].to_numpy()


yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

frame2=fig1.add_axes((1, .1, 2.5, 1))
plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
#plt.savefig('diatomics_'+molecule+'_prediction.pdf')
plt.show()
#%%
total = 0
dict_filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
data_test = dict_filter(molecules, ('H2O', 'CH3F', 'COCl2', 'C2N2', 'H2S', 'C4H2', 'HOBr', 'C2H6', 'CH3I', 'H2', 'CH3Br', 'HO2', 'HNO3', 'NO', 'HCN', 'CH3Cl', 'O2', 'SO2', 'OH', 'NH3', 'CH3CN', 'CO2', 'HBr'))
data_train = dict_filter(molecules, ('CO', 'CS', 'CH3OH', 'O3', 'HCOOH', 'ClO', 'PH3', 'HOCl', 'OCS', 'CH4', 'C2H4', 'SO', 'H2O2', 'COF2', 'N2O', 'H2CO', 'C2H2', 'HCl', 'HC3N', 'NO2', 'HF', 'CS2', 'HI', 'N2'))

data_test = pd.concat([test_data[k] for k in test_data])
data_train = pd.concat([train_data[k] for k in train_data])

data_test = data_test.sample(frac=1)
data_train = data_train.sample(frac=1)
X_train = data_train.drop(['gamma_air'], axis=1)
y_train = data_train['gamma_air']
X_test = data_test.drop(['gamma_air'], axis=1)
y_test = data_test['gamma_air']

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('score = '+str(pipe.score(X_test, y_test)))
total += pipe.score(X_test, y_test)
y_test_plot = y_test.to_numpy()

#print(data_dun.columns)

nu_plot = X_test['J'].to_numpy()


yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

frame2=fig1.add_axes((1, .1, 2.5, 1))
plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
#plt.savefig('diatomics_'+molecule+'_prediction.pdf')
plt.show()
#%%

#%%
data_test = dict_filter(molecules, ('H2O', 'CH3F', 'COCl2', 'C2N2', 'H2S', 'C4H2', 'HOBr', 'C2H6', 'CH3I', 'H2', 'CH3Br', 'HO2', 'HNO3', 'NO', 'HCN', 'CH3Cl', 'O2', 'SO2', 'OH', 'NH3', 'CH3CN', 'CO2', 'HBr'))

for molecule in data_test:
    prediction = data_test[molecule]
    prediction = prediction.sample(frac=1)

    X_test = prediction.drop(['gamma_air'], axis=1)
    y_test = prediction['gamma_air']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    #pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    #pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print('score = '+str(pipe.score(X_test, y_test)))
    total += pipe.score(X_test, y_test)
    y_test_plot = y_test.to_numpy()

    #print(data_dun.columns)

    nu_plot = X_test['J'].to_numpy()


    yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

    print(molecule)
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

    frame2=fig1.add_axes((1, .1, 2.5, 1))
    plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.savefig('diatomics_'+molecule+'_prediction.pdf')
    plt.show()
#%%
total = 0
dict_filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
data_train = dict_filter(molecules, ('H2O', 'CH3F', 'COCl2', 'C2N2', 'H2S', 'C4H2', 'HOBr', 'C2H6', 'CH3I', 'H2', 'CH3Br', 'HO2', 'HNO3', 'NO', 'HCN', 'CH3Cl', 'O2', 'SO2', 'OH', 'NH3', 'CH3CN', 'CO2', 'HBr'))
data_test = dict_filter(molecules, ('CO', 'CS', 'CH3OH', 'O3', 'HCOOH', 'ClO', 'PH3', 'HOCl', 'OCS', 'CH4', 'C2H4', 'SO', 'H2O2', 'COF2', 'N2O', 'H2CO', 'C2H2', 'HCl', 'HC3N', 'NO2', 'HF', 'CS2', 'HI', 'N2'))

data_test = pd.concat([test_data[k] for k in test_data])
data_train = pd.concat([train_data[k] for k in train_data])

data_test = data_test.sample(frac=1)
data_train = data_train.sample(frac=1)
X_train = data_train.drop(['gamma_air'], axis=1)
y_train = data_train['gamma_air']
X_test = data_test.drop(['gamma_air'], axis=1)
y_test = data_test['gamma_air']

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('score = '+str(pipe.score(X_test, y_test)))
total += pipe.score(X_test, y_test)
y_test_plot = y_test.to_numpy()

#print(data_dun.columns)

nu_plot = X_test['J'].to_numpy()


yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

frame2=fig1.add_axes((1, .1, 2.5, 1))
plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
#plt.savefig('diatomics_'+molecule+'_prediction.pdf')
plt.show()
#%%
data_test = dict_filter(molecules, ('CO', 'CS', 'CH3OH', 'O3', 'HCOOH', 'ClO', 'PH3', 'HOCl', 'OCS', 'CH4', 'C2H4', 'SO', 'H2O2', 'COF2', 'N2O', 'H2CO', 'C2H2', 'HCl', 'HC3N', 'NO2', 'HF', 'CS2', 'HI', 'N2'))

for molecule in data_test:
    prediction = data_test[molecule]
    prediction = prediction.sample(frac=1)

    X_test = prediction.drop(['gamma_air'], axis=1)
    y_test = prediction['gamma_air']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    #pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    #pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print('score = '+str(pipe.score(X_test, y_test)))
    total += pipe.score(X_test, y_test)
    y_test_plot = y_test.to_numpy()

    #print(data_dun.columns)

    nu_plot = X_test['J'].to_numpy()


    yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

    print(molecule)
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

    frame2=fig1.add_axes((1, .1, 2.5, 1))
    plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.savefig('diatomics_'+molecule+'_prediction.pdf')
    plt.show()