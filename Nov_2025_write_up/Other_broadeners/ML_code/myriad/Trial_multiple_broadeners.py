### Set up environment

# import packages
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
from glob import iglob
import random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import glob
import sys
from pathlib import Path

# Set parameters
#T = 298      # Kelvin


# define Jeanna's broadening formalism for use later.  Taken from paper ...
def broadening(m, T, ma, mp, b0):
    gamma = 1.7796e-5 * (m/(m-2)) * (1/np.sqrt(T)) * np.sqrt((ma+mp)/(ma*mp)) * b0**2
    return(gamma)


# function does what it says on the tin
def convert_hitran_error_to_uncertainty(gamma_err):
    gamma_err = gamma_err.replace(0, (.5))  # 0  ~~~  unreported or unavailable
    gamma_err = gamma_err.replace(1, (.5))  # 1  ~~~  Default or constant
    gamma_err = gamma_err.replace(2, (.5))  # 2  ~~~  Average or estimate
    gamma_err = gamma_err.replace(3, (.5))  # 3  ~~~  err >= 20 %              50
    gamma_err = gamma_err.replace(4, (.2))  # 4  ~~~  20 >= err >= 10 %        15
    gamma_err = gamma_err.replace(5, (.1))  # 5  ~~~  10 >= err >= 5 %         7.5
    gamma_err = gamma_err.replace(6, (.05))  # 6  ~~~  5 >= err >= 2 %          3.5
    gamma_err = gamma_err.replace(7, (.02))  # 7  ~~~  2 >= err >= 1 %          1.5
    gamma_err = gamma_err.replace(8, (.01))  # 8  ~~~  err <= 1 %               0.5

    return gamma_err


### Read in data

# absolute path to folder containing data
home = str(Path.home())
file_list = glob.glob(home + '/line_broadening/Other_broadeners/Files_with_new_data/*.csv')

# read data files, taking the filename from absolute path
db = {}
for f in file_list:
    print(f)
    
    i = f[67:70].strip('_')
    j = f[72:75].strip('_')

    print(i, j)
    if i == 'Bro' and j == 'eni':
        continue

    file = pd.read_csv(f, low_memory=False, dtype={'J': np.float64, 'Jpp': np.float64}, usecols=['J', 'Ka_aprox', 'Kc_aprox', 'Jpp', 'Kapp_aprox', 'Kcpp_aprox', 'T', 'profile', 'gamma', 'gamma_uncertainty',])
   
    file = file[file['profile'] == 'Voigt']
    if len(file) == 0:
        continue
    
    print(file.head().to_string())
 
    if i + j in db:
        db[i + '_' + j] = pd.concat([db[i + '_' + j], file]).reset_index(drop=True)
    else:
        db[i + '_' + j] = file.reset_index(drop=True)


print("HITRAN!!")

hit_file_list = glob.glob(home + '/line_broadening/model_search/raw_data/*.csv')
hit_db = {}
for f in hit_file_list:
    i = f[52:-4]
    print(i)
    data_i = pd.read_csv(f, low_memory=False, dtype={'J': np.float64, 'Jpp': np.float64})
    gamma_cols = [col for col in data_i.columns if 'gamma' in col if 'err' not in col if 'ref' not in col]
    for broadener in gamma_cols:
        data_ij = data_i.copy()[['J', 'Ka_aprox', 'Kc_aprox', 'Jpp', 'Kapp_aprox', 'Kcpp_aprox', broadener, broadener+'-err']]
        j = broadener.replace('gamma_', '')
        good_data = data_ij[data_ij[broadener]!='#']
        print(j)
        print(len(good_data))
        if len(good_data) == 0:
            continue

        if j == 'self':
            j = i

        good_data['molecule'] = i
        good_data['broadener'] = j
        good_data['T'] = 296
        good_data['profile'] = 'Voigt'
        good_data[broadener+'-err'] = convert_hitran_error_to_uncertainty(good_data[broadener+'-err'])
        good_data = good_data.rename({broadener: 'gamma', broadener+'-err': 'gamma_uncertainty'}, axis=1)
        good_data['gamma'] = good_data['gamma'].astype(float)

        if i + j in db:
            db[i + '_' + j] = pd.concat([db[i + '_' + j], good_data]).reset_index(drop=True)
        else:
            db[i + '_' + j] = good_data.reset_index(drop=True)


small_source_file = pd.read_csv(home + '/line_broadening/Other_broadeners/Files_with_new_data/Broadening_data_from_smaller_sources.csv', dtype={'J':np.float64, 'Jpp':np.float64, 'gamma':np.float64, 'gamma_uncertainty':np.float64}, header=0, index_col=None, usecols=['molecule', 'broadener', 'J', 'Jpp', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox', 'gamma', 'gamma_uncertainty', 'n', 'T', 'profile', 'paper'])
small_source_file = small_source_file.dropna(subset=['molecule', 'broadener', 'J', 'Jpp', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox', 'gamma', 'gamma_uncertainty'])


masters_data = pd.read_csv(home + '/line_broadening/Other_broadeners/Files_with_new_data/LLM_data/LLM_scraped_data.csv', index_col=0)
masters_data = masters_data[['molecule', 'broadener', 'J', 'Jpp', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox', 'gamma', 'gamma_uncertainty', 'n', 'T', 'profile', 'paper']].dropna()



small_source_file = pd.concat([small_source_file, masters_data])



#create unique list of names
actives = small_source_file.molecule.unique()

#create a data frame dictionary to store your data frames
DataFrameDict = {elem : pd.DataFrame() for elem in actives}

for key in DataFrameDict.keys():
    DataFrameDict[key] = small_source_file[:][small_source_file.molecule == key]

dfd3 = {}
for item, data in DataFrameDict.items():
    perts = data.broadener.unique()
    dfd2 = {elem : pd.DataFrame() for elem in perts}
    for key in dfd2.keys():
        dfd2[key] = data[:][data.broadener == key]
        dfd3[item+'_'+key] = dfd2[key]



print('SMALL SOURCES')
for key, data in dfd3.items():
    data = data[data['profile'] == 'Voigt']
    if len(data) == 0:
        continue

    i, j = str.split(key, '_')
    print(i)
    print(j)
    print(data)
    if i+j in db:
        db[i+'_'+j] = pd.concat([db[i+'_'+j], data]).reset_index(drop=True)
    else:
        db[i+'_'+j] = data.reset_index(drop=True)



molecule_parameters = pd.read_csv(home + '/line_broadening/molecule_parameters.csv', index_col=0)

# dictionary of molecules of data - condensed version
molecules = {}

# take only molecules for which there is full data
for key, data in db.items():
    active, perturber = str.split(key, '_')
    print(active)
    print(perturber)
    if perturber == 'Ar':
        continue
    if active == 'CHF3':
        continue
    if active == '13CO':
        continue
    print('~~~~~~~~~~~~~~~')
    print(data.head().to_string())
    if 'local_iso_id' in data.columns:
        data = data[data['local_iso_id'] == 1]

    if active not in molecule_parameters.index:
        continue
    if perturber not in molecule_parameters.index:
        continue

    active_params = molecule_parameters.loc[active]
    pert_params = molecule_parameters.loc[perturber]

    for index, info in active_params.items():
        data['active_' + index] = info
    for index, info in pert_params.items():
        data['perturber_' + index] = info
    data = data[['J', 'Ka_aprox', 'Kc_aprox',
                 'Jpp', 'Kapp_aprox', 'Kcpp_aprox', 'T', 'profile', 'gamma', 'gamma_uncertainty',
                 'active_weight', 'active_dipole', 'active_m', 'active_d',
                 'active_polar', 'active_B0a', 'active_B0b', 'active_B0c',
                 'perturber_weight', 'perturber_dipole', 'perturber_m', 'perturber_d',
                 'perturber_polar', 'perturber_B0a', 'perturber_B0b', 'perturber_B0c']]
    data['m'] = data['active_m'] + data['perturber_m']
    data['d_act_per'] = data['active_d'] / 2 + data['perturber_d'] / 2
    data = data.drop(['active_d', 'active_m', 'perturber_d', 'perturber_m'], axis=1)

    data['fractional_error'] = data['gamma_uncertainty'] / data['gamma']

    print(len(data))
    data = data.dropna()
    print(len(data))
    data = data[data['gamma'] != 0]
    data = data[data['gamma_uncertainty'] != 0]

    branch = data['Jpp'] - data['J']
    data = data.drop(branch[abs(branch) > 2].index)
    print(len(data))
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

    data['M'] = data['P'] + data['Q'] + data['R'] + data['O'] + data['S']
    data = data.drop(columns=['P', 'Q', 'R', 'O', 'S']).reset_index()

    molecules[key] = data
    broadness_jeanna = broadening(data['m'][0], data['T'][0], data['active_weight'][0], data['perturber_weight'][0],
                                  data['d_act_per'][0])
    molecules[key]['broadness_jeanna'] = broadness_jeanna

###  Investigate data

# Work out the average error codes of each molecule.  Count how many of each error code, and how many points there are

# list of molecule names
molecule_names = []
# list of the proportion of data that is code 3 or above for each molecule
error_2s = []
# list of the number of points each molecule contains
datapoints = []
# list of the mean weights of each molecule's data
weights = []

for key, data in molecules.items():
    molecule, broadener = key.split('_')
    molecule_names.append(key)
    x = data['fractional_error']
    error_2s.append(x.ge(1.0).count() / len(x))
    datapoints.append(len(x))

    # take weight as gamma-air-err
    # data=data.sample(frac=1)
    weight = data['fractional_error']
    # Give helpful weightings
    # reweight 0 to tiny, because 0 gives /0 error
    weight22 = (1 / weight) ** 2
    max_weight = 1000
    weight22 = weight22.clip(upper=max_weight)
    # reassign weight into dictionary
    data['weight'] = weight22
    weights += list(weight22)


# Print out how much data there is which is 'good' (out of every datapoint)
e2 = np.array(error_2s)
dat = np.array(datapoints)
good_data = e2 * dat
print('percentage "good" data =')
print(str(100 * sum(good_data) / sum(dat)) + '%')

largest_data = max(datapoints)
total_data = sum(datapoints)
print(total_data)
mean_weight = np.mean(weights)
print('mean weight across molecules = ' + str(mean_weight)) 
### Prepare data for use

new_total_data = 0
# weight data by error code, currently error code = weighting
for key, data in molecules.items():
    active, perturber = str.split(key, '_')

    mean_data_weight = np.mean(data['weight'])
    weighting_for_this_molecule = (largest_data / (len(data) * len(molecules))) * (mean_data_weight / mean_weight) #* (1000000000 / total_data)# * total_weight_errors
    
    oversampling_threshold = 10
    if weighting_for_this_molecule > oversampling_threshold:
        weighting_for_this_molecule = oversampling_threshold

    print(key)
    print(len(data))
    print(weighting_for_this_molecule)
    data = data.sample(frac=weighting_for_this_molecule, replace=True, weights=data['weight'])
    
    print(len(data))
    
    data['key'] = key
    new_total_data += len(data)
    # assign data back to dictionary
    molecules[key] = data

print('data before = '+str(total_data)+' and data after weighting = '+str(new_total_data))

import random

keys = list(molecules.items())
random.seed(41)
random.shuffle(keys)
molecules = dict(keys)

# Dictionary of molecules, and test/training data
molecule_list = {}

# test data set size
n = 5

# collect 'training data' from all other molecules, except the labelled one
i = 0
# collect 'training data' from all other molecules, except the labelled one
for molecule in molecules:
    if not i % n:
        print(i)
        # molecule is being tested
        test_data = {k: molecules[k] for k in list(molecules)[i:i + n]}
        # take test molecule out of dictionary
        train_data = set(molecules) - set(test_data)
        # dictionary of molecules in test data
        train_data = {k: molecules[k] for k in train_data}

        # All test data in one dataframe
        data_test = pd.concat([test_data[k] for k in test_data])

        # Take all train data into one dataframe
        data_train = pd.concat([train_data[k] for k in train_data])

        # add data into molecule_list dictionary
        moles = ','.join(list(test_data.keys()))
        molecule_list[moles] = [data_train, data_test]

    i += 1


plot_data_list = []

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Machine learning!
counter = 0
pipe_container = []

total = 0
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, lognorm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import export_graphviz
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import VotingRegressor

flag = False
for key, data in molecule_list.items():
    print(len(data[1]))
    print(data[1].columns)

    # take out training and test data
    data_train = data[0]
    data_test = data[1]

    # Check if there are null values
    # if data_train.isnull().values.any():
    #    raise ValueError("We're getting null values here, might be good to cut them out")

    # shuffle data, randomised lines of each molecule for machine learning
    print(len(data_train))
    data_test = data_test.sample(frac=1)
    print("lets train!!!")
    data_train = data_train.sample(frac=0.01)

    # print('data splitted')

    # Training data - separate out all x values and y (broadening) values.  gamma-err is weighting
    X_train = data_train.drop(['gamma', 'gamma_uncertainty', 'fractional_error', 'weight', 'profile', 'key'], axis=1)
    y_train = data_train['gamma']
    weight_train = data_train['weight']
    # Separate out test data
    X_test = data_test.drop(['gamma', 'gamma_uncertainty', 'fractional_error', 'weight', 'profile', 'key'], axis=1)
    y_test = data_test['gamma']
    weight_test = data_test['weight']
    key_array = data_test['key']

    # Create pipeline of scaling, then ML method
    pipe = make_pipeline(StandardScaler(), VotingRegressor(
        estimators=[('hist', HistGradientBoostingRegressor()),
                    ('ada', AdaBoostRegressor()),
                    ('svr', SVR()),
                    ('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2)),
                    ('mlp',
                     MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42,
                                  verbose=1, n_iter_no_change=1, tol=0.00001))]
        , n_jobs=-1, verbose=True
    ))

    print("X and Y !!!!!!")
    print(X_train.head().to_string())
    print(y_train.head().to_string())


    pipe.fit(X_train, y_train)  # , gaussianprocessregressor__sample_weight=weight_train)

    # Predict broadening values
    y_pred = pipe.predict(X_test)

    print('leaf nodes = ' + str(pipe))

    # print out the scor
    score = pipe.score(X_test, y_test, sample_weight=weight_test)
    mse_score = mean_squared_error(y_pred, y_test, sample_weight=weight_test)

    print(key + ' has score = ' + str(score))
    print('mean square error = ' + str(mse_score))

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # Add up the total score - allows methods to be compared
    if score > -10:
        total += score

    # Get data into matplotlib friendly form
    y_test_plot = y_test.to_numpy()
    x_plot = X_test['M'].to_numpy()

    plot_data_list.append([key, x_plot[-100000:], y_test_plot[-100000:], y_pred[-100000:], score, mse_score,
                           X_test['active_weight'][-100000:], key_array[-100000:], weight_test[-100000:]])
    pipe_container.append(pipe)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


print('total score = ' + str(total))


date = str(datetime.today().strftime('%Y-%m-%d'))
with open(home+'/Scratch/'+date+'_other_broadeners_plot_data_list.pkl', 'wb') as f:
    pickle.dump(plot_data_list, f)


with open(home+'/Scratch/'+date+'_other_broadeners_pipe_container.pkl', 'wb') as f:
    pickle.dump(pipe_container, f)

