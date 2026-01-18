import pickle

### Set up environment

# import packages
import copy
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
        # data['Mb'] = data['M']/data['B0c']
        # assign data to molecule
        molecules[key] = data



        #print(data)


        # calculate jeanna broadness, and add to dictionary
        #print(data['m'].loc[0])
        #print(T)
        #print(data['molecule_weight'][0])
        #print(data['air_weight'][0])
        #print(data['findair'][0])
        #print(key)
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

# weight data by error code, currently error code = weighting
for molecule, data in molecules.items():
    # take weight as gamma-air-err
    #data=data.sample(frac=1)
    weight = data['gamma_air-err']
    # Give helpful weightings
    # reweight 0 to tiny, because 0 gives /0 error
    
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

 
    average_weight = np.mean(weight)
    points = 1000000000*average_weight/len(data)
    print(molecule)
    print(average_weight)
    print(points) 
    data = data.sample(frac=points, replace=True)
    #data = pd.concat([data]*points)
    # assign data back to dictionary
    molecules[molecule] = data


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
    
    # take out training and test data
    data_train = data[0]
    data_test = data[1]
    
    # shuffle data, randomised lines of each molecule for machine learning
    print(len(data_train))
    data_test = data_test.sample(frac=1)
    data_train = data_train.sample(frac=0.03)
    


    # Training data - separate out all x values and y (broadening) values.  gamma-err is weighting
    X_train = data_train.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_train = data_train['gamma_air']
    weight_train = data_train['gamma_air-err']
    # Separate out test data
    X_test = data_test.drop(['gamma_air', 'gamma_air-err'], axis=1)
    y_test = data_test['gamma_air']
    weight_test = data_test['gamma_air-err']


    # Create pipeline of scaling, then ML method
    pipe = make_pipeline(StandardScaler(), VotingRegressor(
                                               estimators=[('hist', HistGradientBoostingRegressor(learning_rate=1, max_leaf_nodes=None, random_state=42)),
                                                           ('ada', AdaBoostRegressor()),
                                                           ('svr', SVR()),
                                                           ('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2))]
                                                           #('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.00001))]
                                                , n_jobs=-1, verbose=True
                                               ))



    pipe.fit(X_train, y_train)#, mlpregressor__sample_weight=weight_train)
    
    y_pred = pipe.predict(X_test)

    cols_vals_train = []
    #cols_vals_train2= []
    mean_y_train = []
    for weight in X_train['molecule_weight'].unique():
        mean_y_train.append(np.mean(y_train[X_train['molecule_weight'] == weight]))
        cols_vals_train.append(X_train[X_train['molecule_weight']==weight].drop(['J', 'Jpp', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox', 'M'], axis=1).iloc[0])
        #cvt2 = X_train[X_train['molecule_weight']==weight].drop(['J', 'Jpp', 'Ka', 'Kapp', 'Kc', 'Kcpp', 'M'])
        #cvt2
        #cols_vals_train.append(cvt2.iloc[0])
    cols_vals_train = pd.concat(cols_vals_train, axis=1).T
    
    #print(cols_vals_train)
    #print(mean_y_train)


    cols_vals_test = []
    #cols_vals_test2 = []
    mean_y_test = []
    mean_y_pred = []
    for weight in X_test['molecule_weight'].unique():
        myp = np.mean(y_pred[X_test['molecule_weight']==weight])
        mean_y_pred.append(myp)
        mean_y_test.append(np.mean(y_test[X_test['molecule_weight'] == weight]))
        cols_vals_test.append(X_test[X_test['molecule_weight']==weight].drop(['J', 'Jpp', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox', 'M'], axis=1).iloc[0])
        #cvt2 = X_test[X_test['molecule_weight']==weight].drop(['J', 'Jpp', 'Ka', 'Kapp', 'Kc', 'Kcpp', 'M'])
        #cvt2['y_pred'] = myp
        #cols_vals_test2.append(cvt2.iloc[0])
    cols_vals_test = pd.concat(cols_vals_test, axis=1).T
    #cols_vals_test2 = pd.concat(cols_vals_test)

    #print(mean_y_pred, mean_y_test)

    print('leaf nodes = '+str(pipe))


    # print out the scor
    score = pipe.score(X_test, y_test, sample_weight=weight_test)
    mse_score = mean_squared_error(y_pred, y_test, sample_weight=weight_test)
    
    
    print(key+' has score = '+str(score))
    print('mean square error = '+str(mse_score))

    print()
    
    pipe2 = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.000001))
    pipe2.fit(cols_vals_train, mean_y_train)#, mlpregressor__sample_weight=weight_train)
    mean_y_pred_2 = pipe2.predict(cols_vals_test)


    pipe3 = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.000001))
    pipe3.fit(X_train, y_train)#, mlpregressor__sample_weight=weight_train)
    y_pred_3 = pipe3.predict(X_test)
    
    y_pred_final_2 = copy.copy(y_pred)
    y_pred_final_3 = copy.copy(y_pred)
    mean_y_test3 = []
    n=0
    for weight in X_test['molecule_weight'].unique():
        mean_y_pred_3 = np.mean(y_pred_3[X_test['molecule_weight'] == weight])
        #mean_y_pred = np.mean(y_pred[X_test['molecule_weight']==weight])
        y_pred_final_3[X_test['molecule_weight']==weight] = mean_y_pred_3 / mean_y_pred[n] * y_pred[X_test['molecule_weight']==weight]
        y_pred_final_2[X_test['molecule_weight']==weight] = mean_y_pred_2[n] / mean_y_pred[n] * y_pred[X_test['molecule_weight']==weight]
        n+=1



    print(mean_y_pred, mean_y_test, mean_y_pred_2)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    # Add up the total score - allows methods to be compared
    if score > -10:
        total += score
    
    
    # Get data into matplotlib friendly form
    y_test_plot = y_test.to_numpy()
    x_plot = X_test['M'].to_numpy()
    
   
    plot_data_list.append([key, x_plot[-100000:], y_test_plot[-100000:], y_pred[-100000:], score, mse_score, X_test['molecule_weight'][-100000:], y_pred_final_2[-100000:], y_pred_final_3[-100000:]])#, dot_data])#, pipe])

    
    pipe_container.append([pipe, pipe2, pipe3])
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



print('total score = '+str(total))

import pickle 

with open('try_mlp_to_scale_FULL_TRAINING_data.pkl', 'wb') as f:
    pickle.dump(plot_data_list, f)

with open('try_mlp_to_scale_FULL_TRAINING.pkl', 'wb') as f:
    pickle.dump(pipe_container, f)


#with open("baseline.csv", "w") as f:
#    wr = csv.writer(f)
#    wr.writerows(plot_data_list)
'''
with open('baseline_results.json', 'wb') as fp:
    json.dump(plot_data_list, fp)'''
