import pickle

### Set up environment

# import packages
import pandas as pd
import numpy as np
import os
from glob import iglob
#import matplotlib.pyplot as plt
#import seaborn as sns
import random
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
#from matplotlib.pyplot import figure
from sklearn.neural_network import MLPRegressor
import glob

import json
import csv

# Set parameters
T = 298      # Kelvin


# define Jeanna's broadening formalism for use later.  Taken from paper ...
def broadening(m, T, ma, mp, b0):
    gamma = 1.7796e-5 * (m/(m-2)) * (1/np.sqrt(T)) * np.sqrt((ma+mp)/(ma*mp)) * b0**2
    return(gamma)




### Read in data

# import data

# absolute path to folder containing data
#rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/hitran_data/**/*'
# be selective for data files
#file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-10:] == "/1_iso.csv" if "readme" not in f]
#file_list = 

path = ""
file_list = glob.glob(path + 'raw_data/*.csv')




# read data files, taking the filename from absolute path
db = {}
for f in file_list:
    i = f[9:-4]
    #print(i)
    db[i] = pd.read_csv(f, low_memory=False)
    
    


# dictionary of molecules of data - condensed version
molecules = {}

# take only molecules for which there is full data
for key, data in db.items():
    # filter for rotational constant 3D - as that was the last thing added
    if 'B0a' in data.columns:
        #if data['symmetry'][0] != 1:
        #    if data['symmetry'][0] != 4:
        #        continue
        #if 'K' not in data.columns:
        #    data['K'] = np.NaN
        #if 'Ka' not in data.columns:
        #    data['Ka'] = np.NaN
        #    data['Kc'] = np.NaN
        #print(data.columns[50:])
        #if 'v' in data.columns:
        #    data['vee'] = data['v']
        #if 'v1' in data.columns:
        #    data['vee'] = data['v1']
        # take needed parameters
        data = data[['gamma_air', 'J', 'Jpp', 'molecule_weight', 'gamma_air-err', 'm', 'findair', 
                     'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox']]#, 'K', 'Ka', 'Kc']]#, 'symmetry', 'vee']]#'molecule_weight', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox']]#, 'nu', 'sw', 'a']]

        # some data missing J values, just get rid of it.
        data = data.dropna()
        
        
        if key == 'C2H6':
            data = data.drop(data[(data['J'] > 20) & (data['gamma_air'] > 0.08)].index)
        
        
        branch = data['Jpp'] - data['J']
        data = data.drop(branch[abs(branch)>2].index)
        branch = data['Jpp'] - data['J']
        
        data['P'] = -data['Jpp'][branch==1]
        data['Q'] = data['Jpp'][branch==0]
        data['R'] = data['Jpp'][branch==-1] +1
        data['O'] = -data['Jpp'][branch==2]
        data['S'] = data['Jpp'][branch==-2] +1

        data['P'] = data['P'].fillna(0)
        data['Q'] = data['Q'].fillna(0)
        data['R'] = data['R'].fillna(0)
        data['O'] = data['O'].fillna(0)
        data['S'] = data['S'].fillna(0)
        #data = data.fillna(0)

        data['M'] = data['P'] + data['Q'] + data['R'] + data['O'] + data['S']
        data = data.drop(columns=['P', 'Q', 'R', 'O', 'S'])
        #data['M'] = data['M']/data['B0c']
        # assign data to molecule
        molecules[key] = data



        #print(data)


        # calculate jeanna broadness, and add to dictionary
        #print(data['m'].loc[0])
        #print(T)
        #print(data['molecule_weight'][0])
        #print(data['air_weight'][0])
        #print(data['findair'][0])
        if key == 'CH3CN':
            broadness_jeanna = broadening(data['m'][20], T, data['molecule_weight'][20], data['air_weight'][20], data['findair'][20])
        else:
            broadness_jeanna = broadening(data['m'][2], T, data['molecule_weight'][2], data['air_weight'][2], data['findair'][2])
        molecules[key]['broadness_jeanna'] = broadness_jeanna
        #print(molecules[key])
        molecules[key] = molecules[key].drop(columns=['air_weight'])#symmetry
        #print(molecules[key])



###  Investigate data

# Work out the average error codes of each molecule.  Count how many of each error code, and how many points there are

# list of molecule names
molecule_names = []
# list of the proportion of data that is code 3 or above for each molecule
error_2s = []
# list of the number of points each molecule contains
datapoints = []


for key, data in molecules.items():
    #print(key)
    molecule_names.append(key)
    x = data['gamma_air-err']
    # array of a 2s, b 3s, c 4s, etc...
    y = x.value_counts().sort_index()


    # if there is data classed as 2 or below, class it as 'bad'
    if 2 in y.index:
        error_2s.append(1-y.cumsum()[2]/x.count())
        datapoints.append(x.count())
    # otherwise all data is 'good'
    else:
        error_2s.append(1-0)
        datapoints.append(x.count())
        
        

# Print out how much data there is which is 'good' (out of every datapoint)
e2 = np.array(error_2s)
dat = np.array(datapoints)
good_data = e2*dat
print('percentage "good" data =')
print(sum(good_data)/sum(dat))





### Prepare data for use

'''
# reset molecules so that all molecules have the same number of points
for molecule, data in molecules.items():
    # normalise the amount of data compared to the molecule with the most data (SO2)
    points = 549424//len(data)
    #print(len(data))
    # repeat data n times, until each has roughly the same amount of data
    data = pd.concat([data]*points)
    # assign data back to dictionar
    molecules[molecule] = data
'''

# weight data by error code, currently error code = weighting
for molecule, data in molecules.items():
    # take weight as gamma-air-err
    #data=data.sample(frac=1)
    weight = data['gamma_air-err']
    # Give helpful weightings
    # reweight 0 to tiny, because 0 gives /0 error
    
    if molecule in ['COCl2', 'C2H2', 'HOBr', 'H2', 'CH3OH', 'HCOOH', 'HOCl', 'COF2', 'HC3N']:
        weight = (1/1000000000)**2
    elif molecule in ['HNO3', 'O2', 'CS', 'ClO', 'CH3CN', 'H2O2', 'C2H4', 'O3']:
        weight = weight.replace(0, (1/500000000)**2)    # 0  ~~~  unreported or unavailable
        weight = weight.replace(1, (1/20000000)**2)    # 1  ~~~  Default or constant
        weight = weight.replace(2, (1/1000000)**2)    # 2  ~~~  Average or estimate
        weight = weight.replace(3, (1/50)**2)     # 3  ~~~  err >= 20 %              50
        weight = weight.replace(4, (1/15)**2)     # 4  ~~~  20 >= err >= 10 %        15
        weight = weight.replace(5, (1/10)**2)    # 5  ~~~  10 >= err >= 5 %         7.5
        weight = weight.replace(6, (1/10)**2)    # 6  ~~~  5 >= err >= 2 %          3.5
        weight = weight.replace(7, (1/10)**2)    # 7  ~~~  2 >= err >= 1 %          1.5
        weight = weight.replace(8, (1/10)**2)    # 8  ~~~  err <= 1 %               0.5
    elif molecule in ['C2H6']:
        weight = weight.replace(0, (1/500000)**2)    # 0  ~~~  unreported or unavailable
        weight = weight.replace(1, (1/20000)**2)    # 1  ~~~  Default or constant
        weight = weight.replace(2, (1/1000)**2)    # 2  ~~~  Average or estimate
        weight = weight.replace(3, (1/50)**2)     # 3  ~~~  err >= 20 %              50
        weight = weight.replace(4, (1/15000000000000)**2)     # 4  ~~~  20 >= err >= 10 %        15
        weight = weight.replace(5, (1/10)**2)    # 5  ~~~  10 >= err >= 5 %         7.5
        weight = weight.replace(6, (1/10)**2)    # 6  ~~~  5 >= err >= 2 %          3.5
        weight = weight.replace(7, (1/10)**2)    # 7  ~~~  2 >= err >= 1 %          1.5
        weight = weight.replace(8, (1/10)**2)    # 8  ~~~  err <= 1 %               0.5
    elif molecule in ['SO2']:
        weight = weight.replace(0, (1/500000000)**2)    # 0  ~~~  unreported or unavailable
        weight = weight.replace(1, (1/20000000)**2)    # 1  ~~~  Default or constant
        weight = weight.replace(2, (1/1000000)**2)    # 2  ~~~  Average or estimate
        weight = weight.replace(3, (1/50000)**2)     # 3  ~~~  err >= 20 %              50
        weight = weight.replace(4, (1/15)**2)     # 4  ~~~  20 >= err >= 10 %        15
        weight = weight.replace(5, (1/10)**2)    # 5  ~~~  10 >= err >= 5 %         7.5
        weight = weight.replace(6, (1/10)**2)    # 6  ~~~  5 >= err >= 2 %          3.5
        weight = weight.replace(7, (1/10)**2)    # 7  ~~~  2 >= err >= 1 %          1.5
        weight = weight.replace(8, (1/10)**2)    # 8  ~~~  err <= 1 %               0.5
    else:
        weight = weight.replace(0, (1/500000)**2)    # 0  ~~~  unreported or unavailable
        weight = weight.replace(1, (1/20000)**2)    # 1  ~~~  Default or constant
        weight = weight.replace(2, (1/1000)**2)    # 2  ~~~  Average or estimate
        weight = weight.replace(3, (1/50)**2)     # 3  ~~~  err >= 20 %              50
        weight = weight.replace(4, (1/15)**2)     # 4  ~~~  20 >= err >= 10 %        15
        weight = weight.replace(5, (1/10)**2)    # 5  ~~~  10 >= err >= 5 %         7.5
        weight = weight.replace(6, (1/10)**2)    # 6  ~~~  5 >= err >= 2 %          3.5
        weight = weight.replace(7, (1/10)**2)    # 7  ~~~  2 >= err >= 1 %          1.5
        weight = weight.replace(8, (1/10)**2)    # 8  ~~~  err <= 1 %               0.5
    
    # reassign weight into dictionary
    molecules[molecule]['gamma_air-err'] = weight
    


    #print(molecule)
    #print(len(data))
    datas=[]
    if isinstance(weight, float):
        points=0
        data = data.sample(frac=points, replace=True)
        molecules[molecule] = data
    else:
        for weight_value in weight.unique():
            amount_data = len(weight[weight==weight_value])
            #print(weight_value)
            #print(amount_data)
            fraction = amount_data / len(data)
            points = 600000000*weight_value*fraction/len(data)
            #print(fraction)
            #print('fraction kept = '+str(points)) 
            #print()
            datas.append(data[weight==weight_value].sample(frac=points, replace=True))
    
        data = pd.concat(datas)
    # assign data back to dictionary
    molecules[molecule] = data
    
        
        
        
    '''average_weight = np.mean(weight)
    points = 600000000*average_weight/len(data)
    print(molecule)
    print('avg_weighting = '+str(average_weight))
    print('fraction kept = '+str(points)) 
    print()
    data = data.sample(frac=points, replace=True)
    #data = pd.concat([data]*points)
    # assign data back to dictionary
    molecules[molecule] = data
    '''


'''
# reset molecules so that all molecules have the same number of points
for molecule, data in molecules.items():
    # normalise the amount of data compared to the molecule with the most data (SO2)
    print(len(data))
    weight = 950863/len(data)
    #weight = 549424/len(data)
    # assign data back to dictionary
    molecules[molecule]['gamma_air-err'] = data['gamma_air-err']*weight
'''

for molecule, data in molecules.items():
    data.to_csv('data/'+molecule+'.csv')




import random
keys = list(molecules.items())
random.seed(42)
random.shuffle(keys)
molecules = dict(keys)
    
# Dictionary of molecules, and test/training data
molecule_list = {}

# collect 'training data' from all other molecules, except the labelled one
print(molecules.keys())
i=0
# collect 'training data' from all other molecules, except the labelled one
for molecule in molecules:
    if not i%5:
        # molecule is being tested
        test_data = {k: molecules[k] for k in list(molecules)[i:i+5]}
        #data_test = molecules[molecule]
        # take test molecule out of dictionary
        train_data = set(molecules) - set(test_data)
        #print(test_data)
        print(test_data.keys(), train_data)
        # dictionary of molecules in test data
        train_data = {k: molecules[k] for k in train_data}
        
        # All test data in one dataframe
        data_test = pd.concat([test_data[k] for k in test_data])

        # Take all train data into one dataframe
        data_train = pd.concat([train_data[k] for k in train_data])
        
        # add data into molecule_list dictionary
        moles = ','.join(list(test_data.keys()))
        molecule_list[moles] = [data_train, data_test]
        
    i+=1
    



#print(molecule_list)



'''
# concatenate all dataframes once
data_train_all = pd.concat(list(molecules.values()))


molecule_list = {}

a = []
for i, data_test in enumerate(list(molecules.values())):
    # slice the concatenated dataframe
    data_train = pd.concat([data_train_all.iloc[:i*len(molecule_list)], data_train_all.iloc[(i+1)*len(molecule_list):]])

    molecule_list[list(molecules.keys())[i]] = [data_train, data_test]


'''
'''
# Dictionary of molecules, and test/training data
molecule_list = {}

# collect 'training data' from all other molecules, except the labelled one
for molecule in molecules:
    # molecule is being tested
    data_test = molecules[molecule].to_numpy()
    # take test molecule out of dictionary
    train_data = {}
    for k in molecules:
        if k != molecule:
            train_data[k] = molecules[k].to_numpy()

    data_train = np.concatenate([train_data[k] for k in train_data])
    molecule_list[molecule] = [data_train, data_test]
'''



def what_is_error(weight):
    if weight == (1/500000)**2:
        weight = 'Unavailable'
    if weight == (1/20000)**2:
        weight = 'Constant'
    if weight == (1/1000)**2:
        weight = 'Estimate'
    if weight == (1/50)**2:
        weight = '>20%'
    if weight == (1/15)**2:
        weight = '20> >10%'
    if weight == (1/10)**2:
        weight = '<10%'
    return weight



plot_data_list = []
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Machine learning!
#print('Can you get to the start of ML?')

## start machine learning
#total_1oerr2 = 0
#total_1oerr = 0
#total_err = 0
#total_err2 = 0
#total_1orterr = 0
#total_nowt = 0
#total_MLP = 0
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
#>>> from sklearn.pipeline import Pipeline
#>>> import numpy as np
flag = False
#kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
# learn on each molecule
for key, data in molecule_list.items():
    # print out which molecule we're looking at
    #counter += 1
    #if counter < 1:
    #    continue
    
    #distributions = dict(histgradientboostingregressor__loss=['least_squares', 'least_absolute_deviation', 'poisson'],# 'quantile'],
    #                      histgradientboostingregressor__learning_rate=lognorm(2),         histgradientboostingregressor__learning_rate=uniform(loc=0.2, scale=0.9))
    #                      #histgradientboostingregressor__max_iter=np.random.randint(10, 1000, 1000),       #lognorm(1.5, scale=300),
    #                      #histgradientboostingregressor__max_leaf_nodes=np.random.randint(2, 100, 1000),              #uniform(loc=0, scale=300),
    #                      #histgradientboostingregressor__l2_regularization=uniform(loc=0, scale=10)) 
    


    #distributions = dict(histgradientboostingregressor__min_samples_leaf=[1000, 100, 20, 5, 1])
    #distributions = dict(mlpregressor__hidden_layer_sizes=[(100, 100), (1000, 1000), (5000, 500), (500, 5000)], 
    #                     mlpregressor__activation=['tanh', 'relu'],
    #                     mlpregressor__alpha=[0.0001, 0.001, 0.01])
    
    #if key == 'HO2':
    #    flag=True
    #elif key == 'NO':
    #    flag=False
    #print(molecule)
    #if flag == False:
    #    continue
    
    
    # take out training and test data
    data_train = data[0]
    data_test = data[1]
    
    # Check if there are null values
    #if data_train.isnull().values.any():
    #    raise ValueError("We're getting null values here, might be good to cut them out")

    # shuffle data, randomised lines of each molecule for machine learning
    print(len(data_train))
    data_test = data_test.sample(frac=1)
    print("lets train!!!")
    data_train = data_train.sample(frac=0.01)
    

    #print('data splitted')



    # Training data - separate out all x values and y (broadening) values.  gamma-err is weighting
    X_train = data_train.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_train = data_train['gamma_air']
    weight_train = data_train['gamma_air-err']
    #weight_train_1ovsquare = data_train['gamma_air-err']
    #weight_train_1overr = np.sqrt(data_train['gamma_air-err'])
    #weight_train_err = 1/np.sqrt(data_train['gamma_air-err'])
    #weight_train_err_sq = 1/data_train['gamma_air-err']
    #weight_train_1ovsq_root = np.sqrt(np.sqrt(data_train['gamma_air-err']))
    #weight_train = np.sqrt(data_train['gamma_air-err'])
    #weight_train = np.sqrt(data_train['gamma_air-err'])
    # Separate out test data
    X_test = data_test.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_test = data_test['gamma_air']
    weight_test = data_test['gamma_air-err']
    #weight_test_1ovsquare = data_test['gamma_air-err']
    #weight_test_1overr = np.sqrt(data_test['gamma_air-err'])
    #weight_test_err = 1/np.sqrt(data_test['gamma_air-err'])
    #weight_test_err_sq = 1/data_test['gamma_air-err']
    #weight_test_1ovsq_root = np.sqrt(np.sqrt(data_test['gamma_air-err']))

    
    #pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2))

    #FINALLLLL
    
    # Create pipeline of scaling, then ML method
    pipe = make_pipeline(StandardScaler(), VotingRegressor(
                                               estimators=[('hist', HistGradientBoostingRegressor()),
                                                           ('ada', AdaBoostRegressor()),
                                                           ('svr', SVR()),
                                                           ('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2)),
                                                           ('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.00001))]
                                                , n_jobs=-1, verbose=True
                                               ))

    
    '''
    pipe1 = make_pipeline(StandardScaler(), VotingRegressor(
                                               estimators=[#('hist', HistGradientBoostingRegressor(learning_rate=1, max_leaf_nodes=None, random_state=42)),
                                                           ('ada', AdaBoostRegressor()),
                                                           ('svr', SVR()),
                                                           ('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2)),
                                                           ('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.00001))]
                                                , n_jobs=-1, verbose=True
                                               ))
    pipe2 = make_pipeline(StandardScaler(), VotingRegressor(
                                               estimators=[('hist', HistGradientBoostingRegressor(learning_rate=1, max_leaf_nodes=None, random_state=42)),
                                                           #('ada', AdaBoostRegressor()),
                                                           ('svr', SVR()),
                                                           ('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2)),
                                                           ('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.00001))]
                                                , n_jobs=-1, verbose=True
                                               ))
    pipe3 = make_pipeline(StandardScaler(), VotingRegressor(
                                               estimators=[('hist', HistGradientBoostingRegressor(learning_rate=1, max_leaf_nodes=None, random_state=42)),
                                                           ('ada', AdaBoostRegressor()),
                                                           #('svr', SVR()),
                                                           ('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2)),
                                                           ('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.00001))]
                                                , n_jobs=-1, verbose=True
                                               ))
    pipe4 = make_pipeline(StandardScaler(), VotingRegressor(
                                               estimators=[('hist', HistGradientBoostingRegressor(learning_rate=1, max_leaf_nodes=None, random_state=42)),
                                                           ('ada', AdaBoostRegressor()),
                                                           ('svr', SVR()),
                                                           #('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2)),
                                                           ('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.00001))]
                                                , n_jobs=-1, verbose=True
                                               ))
    pipe5 = make_pipeline(StandardScaler(), VotingRegressor(
                                               estimators=[('hist', HistGradientBoostingRegressor(learning_rate=1, max_leaf_nodes=None, random_state=42)),
                                                           ('ada', AdaBoostRegressor()),
                                                           ('svr', SVR()),
                                                           ('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2))]
                                                           #('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.00001))]
                                                , n_jobs=-1, verbose=True
                                               ))



    '''

    #('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.00001))]

    #pipe = make_pipeline(StandardScaler(), HistGradientBoostingRegressor(learning_rate=1, max_leaf_nodes=None, random_state=42, verbose=1))
    #pipe = make_pipeline(StandardScaler(), HistGradientBoostingRegressor(random_state=42, verbose=1))
    #pipe = RandomizedSearchCV(pipe, distributions, n_iter=5, cv=2, verbose=3)
    #pipe = make_pipeline(StandardScaler(), AdaBoostRegressor())
    #pipe = make_pipeline(StandardScaler(),  SGDRegressor(max_iter=1000, tol=1e-3))
    #pipe = make_pipeline(StandardScaler(), DummyRegressor(strategy='mean'))
    #pipe = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=9))
    #pipe = make_pipeline(StandardScaler(), RandomForestRegressor())
    #pipe = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 100), alpha=0.001, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=3, tol=0.00001))#, hidden_layer_sizes=(10, 100), alpha=0.001, learning_rate='adaptive', random_state=42, verbose=1))
    #pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression(fit_intercept=False))
    #pipe = make_pipeline(StandardScaler(), SVR(C=5, epsilon=0.2))
    #pipe = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=0.1))
    #pipe = RandomizedSearchCV(pipe, distributions, n_iter=24, cv=2, verbose=3)

    #print('Pipeline made')
    #print(pipe) 
    #pipe.fit(X_train, y_train, linearregression__sample_weight=weight_train)
    #pipe.fit(X_train, y_train, histgradientboostingregressor__sample_weight=weight_train)
    #pipe.fit(X_train, y_train, sgdregressor__sample_weight=weight_train)
    #pipe.fit(X_train, y_train, dummyregressor__sample_weight=weight_train)
    #pipe.fit(X_train, y_train, adaboostregressor__sample_weight=weight_train)
    #pipe.fit(X_train, y_train, decisiontreeregressor__sample_weight=weight_train)
    #pipe.fit(X_train, y_train, randomforestregressor__sample_weight=weight_train)
    #pipe.fit(X_train, y_train, votingregressor__sample_weight=weight_train)
    pipe.fit(X_train, y_train)#, gaussianprocessregressor__sample_weight=weight_train)
    #pipe.fit(X_train, y_train, mlpregressor__sample_weight=weight_train)
    '''pipe1.fit(X_train, y_train)#, mlpregressor__sample_weight=weight_train)
    pipe2.fit(X_train, y_train)#, mlpregressor__sample_weight=weight_train)
    pipe3.fit(X_train, y_train)#, mlpregressor__sample_weight=weight_train)
    pipe4.fit(X_train, y_train)#, mlpregressor__sample_weight=weight_train)
    pipe5.fit(X_train, y_train)#, mlpregressor__sample_weight=weight_train)
    '''

    # Predict broadening values
    y_pred = pipe.predict(X_test)
    '''y_pred1 = pipe.predict(X_test)
    y_pred2 = pipe.predict(X_test)
    y_pred3 = pipe.predict(X_test)
    y_pred4 = pipe.predict(X_test)
    y_pred5 = pipe.predict(X_test)
    '''

    #print(pipe.best_params_)
    #y_pred2 = pipe2.predict(X_test)
    #y_pred3 = pipe3.predict(X_test)
    #y_pred4 = pipe4.predict(X_test)
    #y_pred5 = pipe5.predict(X_test)
    #y_pred6 = pipe6.predict(X_test)
    #y_pred7 = pipe7.predict(X_test)


    print('leaf nodes = '+str(pipe))

    #print('fitting done')



    # print out the scor
    score = pipe.score(X_test, y_test, sample_weight=weight_test)
    mse_score = mean_squared_error(y_pred, y_test, sample_weight=weight_test)
    
    '''score1 = pipe1.score(X_test, y_test, sample_weight=weight_test)
    score2 = pipe2.score(X_test, y_test, sample_weight=weight_test)
    score3 = pipe3.score(X_test, y_test, sample_weight=weight_test)
    score4 = pipe4.score(X_test, y_test, sample_weight=weight_test)
    score5 = pipe5.score(X_test, y_test, sample_weight=weight_test)
    '''
    #score2 = pipe2.score(X_test, y_test, weight_test_1ovsquare)
    #score3 = pipe3.score(X_test, y_test, weight_test_1ovsquare)
    #score4 = pipe4.score(X_test, y_test, weight_test_1ovsquare)
    #score5 = pipe5.score(X_test, y_test, weight_test_1ovsquare)
    #score6 = pipe6.score(X_test, y_test, weight_test_1ovsquare)
    #score7 = pipe7.score(X_test, y_test, weight_test_1ovsquare)
    
    print(key+' has score = '+str(score))
    print('mean square error = '+str(mse_score))

    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    '''print('minor scores = ')
    print(score1, score2, score3, score4, score5)
    '''

    #print('score 1/err = '+str(score2))
    #print('score err = '+str(score3))
    #print('score err2 = '+str(score4))
    #print('score 1/rt(err) = '+str(score5))
    #print('score nowt = '+str(score6))
    #print('score MLP = '+str(score7))
    
    # Add up the total score - allows methods to be compared
    if score > -10:
        total += score
    #total_1oerr2 += score1
    #total_1oerr += score2
    #total_err += score3
    #total_err2 += score4
    #total_1orterr += score5
    #total_nowt += score6
    #total_MLP = score7
    
    
    # Get data into matplotlib friendly form
    y_test_plot = y_test.to_numpy()
    x_plot = X_test['M'].to_numpy()


    # prepare to plot different accuracy different colour
    #err_codes = data_test['gamma_air-err'].value_counts().sort_index()
    #data_by_vib_lev = {}
    
    '''
    cv_results = pd.DataFrame(pipe.cv_results_).sort_values("mean_test_score", ascending=False)
    column_results = [f"param_{name}" for name in distributions.keys()]
    column_results += ["mean_test_score", "std_test_score", "rank_test_score"]
    cv_results = cv_results[column_results]
    def shorten_param(param_name):
        if "__" in param_name:
            return param_name.rsplit("__", 1)[1]
        return param_name
    cv_results = cv_results.rename(shorten_param, axis=1)
    pivoted_cv_results = cv_results.pivot_table(values="mean_test_score", index=["hidden_layer_sizes"], columns=["alpha"])
    print('CV results:')
    print(cv_results)
    print('LR vs l2')
    print(pivoted_cv_results)
    '''
   
    #plot_data_list.append([key, x_plot[-100000:], y_test_plot[-100000:], y_pred[-100000:], [score, score1, score2, score3, score4, score5], mse_score, X_test['molecule_weight'][-100000:]])#, dot_data])#, pipe])
    plot_data_list.append([key, x_plot[-100000:], y_test_plot[-100000:], y_pred[-100000:], score, mse_score, X_test['molecule_weight'][-100000:]])#, dot_data])#, pipe])
    

    
    #print(pipe[1].train_score_)
    
    #pipe_container.append([pipe, pipe1, pipe2, pipe3, pipe4, pipe5])
    pipe_container.append(pipe)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



print('total score = '+str(total))

import pickle 

with open('final_10-25_voter_data.pkl', 'wb') as f:
    pickle.dump(plot_data_list, f)

with open('final_10-25_voter_forest.pkl', 'wb') as f:
    pickle.dump(pipe_container, f)


#with open("baseline.csv", "w") as f:
#    wr = csv.writer(f)
#    wr.writerows(plot_data_list)
'''
with open('baseline_results.json', 'wb') as fp:
    json.dump(plot_data_list, fp)'''
