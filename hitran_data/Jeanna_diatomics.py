#%%
import pandas as pd
import numpy as np

co = pd.read_csv("CO/1_iso.csv")
#o2 = pd.read_csv("O2/O2_1_iso.csv")
no = pd.read_csv("NO/1_iso.csv")
oh = pd.read_csv("OH/1_iso.csv")
hf = pd.read_csv("HF/1_iso.csv")
hcl = pd.read_csv("HCl/1_iso.csv")
hbr = pd.read_csv("HBr/1_iso.csv")
hi = pd.read_csv("HI/1_iso.csv")
clo = pd.read_csv("ClO/1_iso.csv")
#n2 = pd.read_csv("N2/N2_1_iso.csv")
#h2 = pd.read_csv("H2/H2_1_iso.csv")
cs = pd.read_csv("CS/1_iso.csv")
so = pd.read_csv("SO/1_iso.csv")

#%%
co['Lambda'] = 0
hf['Lambda'] = 0
hbr['Lambda'] = 0
hcl['Lambda'] = 0
hi['Lambda'] = 0
cs['Lambda'] = 0

co['Lambdapp'] = 0
hf['Lambdapp'] = 0
hbr['Lambdapp'] = 0
hcl['Lambdapp'] = 0
hi['Lambdapp'] = 0
cs['Lambdapp'] = 0

co['Omega'] = 0
hf['Omega'] = 0
hbr['Omega'] = 0
hcl['Omega'] = 0
hi['Omega'] = 0
cs['Omega'] = 0

co['Omegapp'] = 0
hf['Omegapp'] = 0
hbr['Omegapp'] = 0
hcl['Omegapp'] = 0
hi['Omegapp'] = 0
cs['Omegapp'] = 0

co['S'] = 0
hf['S'] = 0
hbr['S'] = 0
hcl['S'] = 0
hi['S'] = 0
cs['S'] = 0

co['Spp'] = 0
hf['Spp'] = 0
hbr['Spp'] = 0
hcl['Spp'] = 0
hi['Spp'] = 0
cs['Spp'] = 0

so['Omega'] = 0
so['Omegapp'] = 0
#%%
molecules = {"CO": co, "NO": no, "OH": oh, "HF": hf, "HCl": hcl, "HBr": hbr, "HI": hi, "ClO": clo, "CS": cs, "SO": so}
molecule_list = {"CO": co, "NO": no, "OH": oh, "HF": hf, "HCl": hcl, "HBr": hbr, "HI": hi, "ClO": clo, "CS": cs, "SO": so}
#%%
for molecule in molecules:
    #molecules[molecule] = molecules[molecule].drop('Unnamed: 0', axis=1)
 
    print(molecule)
    print(molecules[molecule].columns)
    
#%%
for molecule in molecules:
    df = molecules[molecule]
    print(molecule)
    #print(df['gamma_air-err'])
    #df = df[df['gamma_air-err']>=3]
    #print(df['gamma_air-err'])
    #print(len(df[df['gamma_air-err']<=2]))
    #if len(df[df['gamma_air-err']<=2]) > 0:
    #    df = df.drop(df[df['gamma_air-err']<=2].index, inplace=True)
    molecules[molecule] = df

#%%
#molecules.pop('SO')
#molecules.pop('CS')
#%%
for molecule in molecules:
    df = molecules[molecule]
    df = df.drop('Unnamed: 0', axis=1)

    molecules[molecule] = df
#%%
x = oh['gamma_air-err'] 
x.value_counts()

print('percentage of data with error code present =')
print(100*x.count()/len(x))
print('percentage of data with error code 2 =')
print(100*x.value_counts()[2]/x.count())
#%%
moleys = []
error_2s = []
datapoints = []

for molecule in molecules:
    df = molecules[molecule]
    x = df['gamma_air-err']
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
import random

for molecule in molecules:
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
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#%%
print(molecule_list['CO'][1].iloc[:, 50:])
#%%
print(molecule_list['CO'][0].columns)
#%%
    #print(data[0][['Fnuc']].count()/len(data[0][['Fnuc']]))
    #print(data[0][['N']].value_counts())
#%%
for molecule in molecules:
    print(molecule)
    data = molecule_list[molecule]

    data_train = data[0][['nu', 'sw', 'a', 'gamma_air', 'v', 'J', 'vpp', 'Jpp', 'molecule_weight', 'air_weight', 'molecule_dipole', 'B0', 'coord', 'polar', 'open_shell', 'wexe', 'mass_ratio', 'S', 'Spp', 'Omega', 'Omegapp', 'Lambda', 'Lambdapp', 'gamma_air-err']]
    data_test = data[1][['nu', 'sw', 'a', 'gamma_air', 'v', 'J', 'vpp', 'Jpp', 'molecule_weight', 'air_weight', 'molecule_dipole', 'B0', 'coord', 'polar', 'open_shell', 'wexe', 'mass_ratio', 'S', 'Spp', 'Omega', 'Omegapp', 'Lambda', 'Lambdapp', 'gamma_air-err']]

    points = 251898//len(data_test)
    print(points)

    data_test = pd.concat([data_test]*points)
    print(len(data_test))
    molecule_list[molecule][0] = data_train
    molecule_list[molecule][1] = data_test
#%%
total = 0

for molecule in molecule_list:
    if molecule in ['HCl', 'CO']:
        data = molecule_list[molecule]
        data_train = data[0]
        data_test = data[1]

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





        #figure(figsize=((15, 7)), dpi=500)
        print(molecule)
        Air = 'Air'
        import math

        fig1 = plt.figure(1)
        print(molecule)
        frame1=fig1.add_axes((1, 1.1, 2.5, 1))
        plt.plot(nu_plot[:1000], y_test_plot[:1000], 'rx', label="Pressure broadening given by HITRAN")
        plt.plot( nu_plot[:1000], y_pred[:1000], 'bx', label="Predicted pressure broadening")
        plt.ylabel(r'Pressure Broadening, $\gamma_{Air}$ / $cm^{-1}atm^{-1}$')
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

#%%
for molecule in molecule_list:
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



    pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0))
    pipe.fit_transform(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print('score = '+str(pipe.score(X_test, y_test)))

    y_test_plot = y_test.to_numpy()

    #print(data_dun.columns)

    nu_plot = X_test['J'].to_numpy()


    yerr = (y_pred - y_test_plot)/np.std(y_test_plot)


    print(molecule)
    print(len(data_test))
    #fig1 = plt.figure(1)
    #frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    #plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx', label=molecule)

    #frame2=fig1.add_axes((1, .1, 2.5, 1))
    #plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.show()
#%%
for molecule in molecule_list:
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



    pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0))
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print('score = '+str(pipe.score(X_test, y_test)))

    y_test_plot = y_test.to_numpy()

    #print(data_dun.columns)

    nu_plot = X_test['J'].to_numpy()


    yerr = (y_pred - y_test_plot)/np.std(y_test_plot)


    print(molecule)
    print(len(data_test))
    #fig1 = plt.figure(1)
    #frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    #plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx', label=molecule)

    #frame2=fig1.add_axes((1, .1, 2.5, 1))
    #plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
    #plt.show()
#%%
import warnings
warnings.filterwarnings('ignore')
#%%

from sklearn.inspection import permutation_importance

feature_importance = pipe.steps[1][1].feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(X_test.columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    pipe.steps[1][1], X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(X_test.columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()
#%%
print(molecule_list.keys())

keys = list(molecule_list.keys())
random.shuffle(keys)

ShuffledStudentDict = dict()
for key in keys:
    ShuffledStudentDict.update({key: molecule_list[key]})
molecule_list = ShuffledStudentDict
print(molecule_list.keys())
#%%

#%%

#%%
print('B0, we, coord, '+str(total))
#%%
print('B0, we, '+str(total))
#%%
print('B0, '+str(total))
#%%
print('B0, coord '+str(total))
#%%
print('B0, we, coord, open_shell '+str(total))
#%%
print('B0, we, coord, open_shell, polar, '+str(total))
#%%
print('B0, we, coord, open_shell, polar, wexe '+str(total))
#%%
print('B0, we, coord, open_shell, polar, wexe, mass_ratio '+str(total))
#%%
print('B0, we, coord, open_shell, polar, wexe, mass_ratio, lambda '+str(total))

#%%
print('B0, we, coord, open_shell, polar, wexe, mass_ratio, lambda, s '+str(total))
#%%
print('B0, we, coord, open_shell, polar, wexe, mass_ratio, lambda, s, omega '+str(total))
#%%
print('B0, we, coord, open_shell, polar, wexe, mass_ratio, s, omega '+str(total))
#%%
