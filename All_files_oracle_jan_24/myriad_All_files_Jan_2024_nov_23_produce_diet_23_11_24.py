import pickle

### Set up environment

# import packages
import pandas as pd
import numpy as np
#from glob import iglob
#import matplotlib.pyplot as plt
#import seaborn as sns
import random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
#from matplotlib.pyplot import figure
from sklearn.neural_network import MLPRegressor
import glob
import sys
#import json
#import csv
from pathlib import Path


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

home = str(Path.home())
file_list = glob.glob(home+'/line_broadening/model_search/raw_data/*.csv')



# read data files, taking the filename from absolute path
db = {}
for f in file_list:
    i = f[52:-4]
    #print(i)
    db[i] = pd.read_csv(f, low_memory=False).reset_index(drop=True)
    
    

# dictionary of molecules of data - condensed version
molecules = {}

# take only molecules for which there is full data
for key, data in db.items():
    # filter for rotational constant 3D - as that was the last thing added
    if 'B0a' in data.columns:
        print(key)
        print(data)
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
        elif key == 'SO3':
            broadness_jeanna = broadening(data['m'][258], T, data['molecule_weight'][258], data['air_weight'][258], data['findair'][258])
        else:
            print(data)
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

# Take all train data into one dataframe
data_train = pd.concat([molecules[k] for k in molecules])
        
    

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
    
    
# take out training and test data
#data_train = data[0]
    

# shuffle data, randomised lines of each molecule for machine learning
data_train = data_train.sample(frac=1)

    

# Training data - separate out all x values and y (broadening) values.  gamma-err is weighting
X_train = data_train.drop(['gamma_air', 'gamma_air-err'], axis=1)
y_train = data_train['gamma_air']
weight_train = data_train['gamma_air-err']
    
# Create pipeline of scaling, then ML method
pipe = make_pipeline(StandardScaler(), VotingRegressor(
                                               estimators=[('hist', HistGradientBoostingRegressor()),
                                                           ('ada', AdaBoostRegressor()),
                                                           ('svr', SVR()),
                                                           ('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, verbose=2)),
                                                           ('mlp', MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42, verbose=1, n_iter_no_change=1, tol=0.00001))]
                                                , n_jobs=-1, verbose=True
                                               ))

    
pipe.fit(X_train, y_train)#, gaussianprocessregressor__sample_weight=weight_train)

# Predict broadening values
#y_pred = pipe.predict(X_test)


print('leaf nodes = '+str(pipe))


print()
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
   
    
pipe_container.append(pipe)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


with open(home+'/Scratch/diet_file_pipeline_2025_04_30.pkl', 'wb') as f:
    pickle.dump(pipe_container, f)

#with open("baseline.csv", "w") as f:
#    wr = csv.writer(f)
#    wr.writerows(plot_data_list)
'''
with open('baseline_results.json', 'wb') as fp:
    json.dump(plot_data_list, fp)'''
