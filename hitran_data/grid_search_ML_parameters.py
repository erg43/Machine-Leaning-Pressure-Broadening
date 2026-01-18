

### Set up environment

# import packages
import pandas as pd
import numpy as np
import os
from glob import iglob
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from matplotlib.pyplot import figure
from sklearn.neural_network import MLPRegressor


# Set parameters
T = 298      # Kelvin


# define Jeanna's broadening formalism for use later.  Taken from paper ...
def broadening(m, T, ma, mp, b0):
    gamma = 1.7796e-5 * (m/(m-2)) * (1/np.sqrt(T)) * np.sqrt((ma+mp)/(ma*mp)) * b0**2
    return(gamma)




### Read in data

# import data

# absolute path to folder containing data
rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/hitran_data/**/*'
# be selective for data files
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-10:] == "/1_iso.csv" if "readme" not in f]

# read data files, taking the filename from absolute path
db = {}
for f in file_list:
    i = f[76:-10]
    db[i] = pd.read_csv(f)
    
    


# dictionary of molecules of data - condensed version
molecules = {}

# take only molecules for which there is full data
for key, data in db.items():
    # filter for rotational constant 3D - as that was the last thing added
    if 'B0a' in data.columns:
        


        # take needed parameters
        data = data[['gamma_air', 'J', 'Jpp', 'molecule_weight', 'gamma_air-err', 'm', 'findair', 
                     'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox']]#, 'nu', 'sw', 'a']]

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
        #data = data.drop(columns=['P', 'Q', 'R', 'O', 'S'])
        # assign data to molecule
        molecules[key] = data

        # calculate jeanna broadness, and add to dictionary
        broadness_jeanna = broadening(data['m'][0], T, data['molecule_weight'][0], data['air_weight'][0], data['findair'][0])
        molecules[key]['broadness_jeanna'] = broadness_jeanna




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

# reset molecules so that all molecules have the same number of points
for molecule, data in molecules.items():
    # normalise the amount of data compared to the molecule with the most data (SO2)
    points = 549425//len(data)

    # repeat data n times, until each has roughly the same amount of data
    data = pd.concat([data]*points)
    # assign data back to dictionary
    molecules[molecule] = data


# weight data by error code, currently error code = weighting
for molecule, data in molecules.items():
    # take weight as gamma-air-err
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
    
    
    
# Dictionary of molecules, and test/training data
molecule_list = {}

# collect 'training data' from all other molecules, except the labelled one
for molecule in molecules:
    # molecule is being tested
    data_test = molecules[molecule]
    # take test molecule out of dictionary
    train_data = set(molecules) - {molecule}

    # dictionary of molecules in test data
    train_data = {k: molecules[k] for k in train_data}

    # Take all train data into one dataframe
    data_train = pd.concat([train_data[k] for k in train_data])

    # add data into molecule_list dictionary
    molecule_list[molecule] = [data_train, data_test]



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




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Machine learning!


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

from sklearn.neural_network import MLPRegressor

# learn on each molecule
for molecule, data in molecule_list.items():
    # print out which molecule we're looking at
    counter += 1
    if counter < 1:
        continue

    #print(molecule)
    
    # take out training and test data
    data_train = data[0]
    data_test = data[1]
    
    # Check if there are null values
    #if data_train.isnull().values.any():
    #    raise ValueError("We're getting null values here, might be good to cut them out")

    # shuffle data, randomised lines of each molecule for machine learning
    data_test = data_test.sample(frac=1)
    data_train = data_train.sample(frac=1)
    
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


    # Create pipeline of scaling, then ML method
    pipe = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
    #pipe2 = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
    #pipe3 = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
    #pipe4 = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
    #pipe5 = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
    #pipe6 = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
    #pipe7 = make_pipeline(StandardScaler(), MLPRegressor())
    
    # Do ML!
    pipe.fit(X_train, y_train, histgradientboostingregressor__sample_weight=weight_train)
    #pipe2.fit(X_train, y_train, histgradientboostingregressor__sample_weight=weight_train_1overr)
    #pipe3.fit(X_train, y_train, histgradientboostingregressor__sample_weight=weight_train_err)
    #pipe4.fit(X_train, y_train, histgradientboostingregressor__sample_weight=weight_train_err_sq)
    #pipe5.fit(X_train, y_train, histgradientboostingregressor__sample_weight=weight_train_1ovsq_root)
    #pipe6.fit(X_train, y_train)#, histgradientboostingregressor__sample_weight=weight_train)
    #pipe7.fit(X_train, y_train)#, histgradientboostingregressor__sample_weight=weight_train)
    # Predict broadening values
    y_pred = pipe.predict(X_test)
    #y_pred2 = pipe2.predict(X_test)
    #y_pred3 = pipe3.predict(X_test)
    #y_pred4 = pipe4.predict(X_test)
    #y_pred5 = pipe5.predict(X_test)
    #y_pred6 = pipe6.predict(X_test)
    #y_pred7 = pipe7.predict(X_test)

    # print out the scor
    score = pipe.score(X_test, y_test, weight_test)
    #score2 = pipe2.score(X_test, y_test, weight_test_1ovsquare)
    #score3 = pipe3.score(X_test, y_test, weight_test_1ovsquare)
    #score4 = pipe4.score(X_test, y_test, weight_test_1ovsquare)
    #score5 = pipe5.score(X_test, y_test, weight_test_1ovsquare)
    #score6 = pipe6.score(X_test, y_test, weight_test_1ovsquare)
    #score7 = pipe7.score(X_test, y_test, weight_test_1ovsquare)
    
    print(key+' has score = '+str(score))
    #print('score 1/err = '+str(score2))
    #print('score err = '+str(score3))
    #print('score err2 = '+str(score4))
    #print('score 1/rt(err) = '+str(score5))
    #print('score nowt = '+str(score6))
    #print('score MLP = '+str(score7))
    
    # Add up the total score - allows methods to be compared
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


    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig1 = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(x_plot[-1000:], y_test_plot[-1000:], 'x', label='HITRAN $\gamma$ data')
    plt.plot(x_plot[:1000], y_pred[:1000], 'o', label="Predicted $\gamma$")
    #plt.plot(x_plot[:1000], y_pred2[:1000], 'o', label="Predicted $\gamma$ - 1ov")
    #plt.plot(x_plot[:1000], y_pred3[:1000], 'o', label="Predicted $\gamma$ - err")
    #plt.plot(x_plot[:1000], y_pred4[:1000], 'o', label="Predicted $\gamma$ - sqerr")
    #plt.plot(x_plot[:1000], y_pred5[:1000], 'o', label="Predicted $\gamma$ - 1ovsqrt")
    #plt.plot(x_plot[:1000], y_pred6[:1000], 'o', label="Predicted $\gamma$ - noweight")
    #plt.plot(x_plot[:1000], y_pred7[:1000], 'o', label="Predicted $\gamma$ - mlpreg")
    Air = 'Air'
    #plt.axhline(y=X_test.iloc[0]['broadness_jeanna'], linestyle='-', label='Jeanna broadening for 298K')
    plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {key}')
    plt.xlabel('M, rotational quantum number')
    plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
    #plt.ylim(0)
    plt.legend()
    plt.savefig(f'{key}_ML_results_trial_1.png')
    
    
    pipe_container.append(pipe)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



print('total score = '+total)