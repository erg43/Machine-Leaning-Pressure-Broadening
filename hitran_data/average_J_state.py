#%% md
# # Set up environment
#%%
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
#%%
# Set parameters

T = 298      # Kelvin
#%%
# define Jeanna's broadening formalism for use later.  Taken from paper ...

def broadening(m, T, ma, mp, b0):
    gamma = 1.7796e-5 * (m/(m-2)) * (1/np.sqrt(T)) * np.sqrt((ma+mp)/(ma*mp)) * b0**2
    return(gamma)
#%% md
# # Read in data
#%%
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
#%%
# get rid of SO3 - it's missing something, and I'm not sure what
#del db['SO3']


# dictionary of molecules of data - condensed version
molecules = {}

# take only molecules for which there is full data
for key, data in db.items():
    # filter for rotational constant 3D - as that was the last thing added
    if 'findair' in data.columns:
        # take needed parameters
        #data = data[['sw', 'gamma_air', 'J', 'Jpp', 'molecule_weight', 'gamma_air-err', 'm', 'findair', 
        #             'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight']] # + 'nu', 'sw', 'a'
        
        # some data missing J values, just get rid of it.
        #data = data.dropna()
        
        
        
        
        # assign data to molecule
        molecules[key] = data
        # calculate jeanna broadness, and add to dictionary
        broadness_jeanna = broadening(data['m'][0], T, data['molecule_weight'][0], data['air_weight'][0], data['findair'][0])
        molecules[key]['broadness_jeanna'] = broadness_jeanna
    

#%%

#%%
def weighted_average(dataframe, value, weight):
    val = dataframe[value]
    wt = dataframe[weight]
    return (val * wt).sum() / wt.sum()
#%%
def what_is_error(weight):
    if weight == 0:
        weight = 'Unavailable'
    if weight == 1:
        weight = 'Constant'
    if weight == 2:
        weight = 'Estimate'
    if weight == 3:
        weight = '>20%'
    if weight == 4:
        weight = '20> >10%'
    if weight == 5:
        weight = '10> >5%'
    if weight == 6:
        weight = '5> >2%'
    if weight == 7:
        weight = '2> >1%'
    if weight == 8:
        weight = '<1%'
    return weight
  
#%%

tab1 = []
tab2 = []
tab3 = []
name = []
difference = []
hiterr = []
percentage_difference = []

for molecule, data in molecules.items():

        
    
    data = data.sample(frac=1)


    average_J = round(weighted_average(data, 'J', 'sw'), 0)

    
    average_broadening = weighted_average(data, 'gamma_air', 'sw')

    broadening_J = data.iloc[0]['broadness_jeanna']
    error = round(data['gamma_air-err'].sum()/len(data['gamma_air-err']))
    
    
    

    tab1.append(int(average_J))
    tab2.append(round(average_broadening, 3))
    tab3.append(round(broadening_J, 3))
    name.append(molecule)
    difference.append(round(-average_broadening + broadening_J, 3))
    percentage_difference.append(str(round((-average_broadening + broadening_J)/average_broadening * 100))+'%')
    hiterr.append(what_is_error(error))


#%%
d = {"Average J of transition":tab1, "Average HITRAN broadening $/cm^(-1)$":tab2, "Jeanna's predicted broadening $/cm^(-1)$":tab3, "Difference in broadening $/cm^(-1)$":difference, 'percentage difference':percentage_difference, 'hiterr':hiterr} 
    

table_comparison = pd.DataFrame(data=d, index=name)
#%%
table_comparison
#%%
from pandas.plotting import table # EDIT: see deprecation warnings below

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

table(ax, table_comparison)  # where df is your data frame

plt.savefig('mytable.png')
#%%
'''import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]

table_comparison.style.apply(background_gradient,
               subset=['Average HITRAN broadening', 'Jeanna broadening', 'difference'],
               cmap='PuBu',
               m=table_comparison['Average HITRAN broadening'].values.min().min(),
               M=table_comparison['Average HITRAN broadening'].values.max().max(),
               low=0,
               high=0.1)

#cm = sns.light_palette('green', as_cmap=True)
#s = table_comparison.style.background_gradient(subset=['Average HITRAN broadening', 'Jeanna broadening', 'difference'], cmap='Greens')
#s'''
#%%
for molecule, data in molecules.items():

    print(molecule)
    
    # Check if there are null values
    #if data.isnull().values.any():
    #    raise ValueError("We're getting null values here, might be good to cut them out")

    # shuffle data, randomised lines of each molecule for machine learning 
    data = data.sample(frac=1)
    
    
    #print(data.head())
    #print(data.shape)
    #data = data.loc[data['J'].dropna()]
    #print(data.shape)
    #print(data.head())

    most_comm_J = round(data.loc[data['sw'].idxmax()]['J'])
    print('most intense transition is at J = '+str(most_comm_J))
    average_J = round(weighted_average(data, 'J', 'sw'), 1)
    print('average J state = '+str(average_J))
    
    difference_mean_mode = round(average_J - most_comm_J, 1)
    print('The average J state at 298K is '+str(difference_mean_mode)+' J higher than the most populous state')
    
    average_broadening = weighted_average(data, 'gamma_air', 'sw')


    broadening_J = data.iloc[0]['broadness_jeanna']
#%%
for molecule, data in molecules.items():

    print(molecule)
    
    # Check if there are null values
    #if data.isnull().values.any():
    #    raise ValueError("We're getting null values here, might be good to cut them out")

    # shuffle data, randomised lines of each molecule for machine learning
    data = data.sample(frac=1)
    
    
    most_comm_J = round(data.loc[data['sw'].idxmax()]['J'])
    print('most intense transition is at J = '+str(most_comm_J))
    average_J = round(weighted_average(data, 'J', 'sw'), 1)
    print('average J state = '+str(average_J))
    
    difference_mean_mode = round(average_J - most_comm_J, 1)
    print('The average J state at 298K is '+str(difference_mean_mode)+' J higher than the most populous state')
    
    
    print()
    print()
    

#%%
for molecule, data in molecules.items():

    print(molecule)
    
    # Check if there are null values
    #if data.isnull().values.any():
    #    raise ValueError("We're getting null values here, might be good to cut them out")

    # shuffle data, randomised lines of each molecule for machine learning 
    data = data.sample(frac=1)
    
    
    #print(data.head())
    #print(data.shape)
    #data = data.loc[data['J'].dropna()]
    #print(data.shape)
    #print(data.head())

    most_comm_J = round(data.loc[data['sw'].idxmax()]['J'])
    print('most intense transition is at J = '+str(most_comm_J))
    average_J = round(weighted_average(data, 'J', 'sw'), 1)
    print('average J state = '+str(average_J))
    
    difference_mean_mode = round(average_J - most_comm_J, 1)
    print('The average J state at 298K is '+str(difference_mean_mode)+' J higher than the most populous state')
    
    average_broadening = weighted_average(data, 'gamma_air', 'sw')
    
    
    
    error = what_is_error(round(data['gamma_air-err'].sum()/len(data['gamma_air-err'])))
    print('average error value is '+error)
    
    
    # Get data into matplotlib friendly form
    y_plot = data['gamma_air'].to_numpy()
    x_plot = data['J'].to_numpy()
    broadening_J = data.iloc[0]['broadness_jeanna']
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig1 = plt.figure(1)
  
  
    
    err_codes = data['gamma_air-err'].value_counts().sort_index()
    data_by_vib_lev = {}

    for code in err_codes.index:
        data_level_x = data[data['gamma_air-err']==code]
        label = str(what_is_error(code))
        plt.plot(data_level_x['J'][-1000:], data_level_x['gamma_air'][-1000:], 'x', label='hitran $\gamma$ data, accuracy = '+label)

  
  
  
  
  
    #plt.plot(x_plot[-1000:], y_plot[-1000:], 'x', label='HITRAN $\gamma$ data')
    plt.plot()
    plt.axvline(x=average_J, linestyle='-', color='k', label='Intensity averaged J state at 296K')
    plt.axhline(y=broadening_J, linestyle='-', color='r', label='$\gamma$ predicted by the Buldyreva formalism')
    plt.axhline(y=average_broadening, linestyle='-', color='g', label='average $\gamma$ from HITRAN data')
    
    '''
        
    
    if 'Ka' in data.columns:
        weighted_ka = round(weighted_average(data, 'Ka', 'sw'), 0)
        
        if 'Kc' in data.columns:
            weighted_kc = round(weighted_average(data, 'Kc', 'sw'), 0)
        
        
            data_at_weighted_K = data.loc[data['Ka']==weighted_ka]
            data_at_weighted_K = data_at_weighted_K.loc[data_at_weighted_K['Kc']==weighted_kc]
            y_k = data_at_weighted_K['gamma_air'].to_numpy()
            x_k = data_at_weighted_K['J'].to_numpy()
            plt.plot(x_k, y_k, 'rx', label='$\gamma$ at average ka values')
            
            
    if 'K' in data.columns:
        weighted_k = round(weighted_average(data, 'K', 'sw'), 0)
        

        data_at_weighted_K = data.loc[data['K']==weighted_k]
        y_k = data_at_weighted_K['gamma_air'].to_numpy()
        x_k = data_at_weighted_K['J'].to_numpy()
        plt.plot(x_k, y_k, 'rx', label='$\gamma$ at average k values')
    
    '''

    plt.xlabel('J, rotational quantum number')
    plt.ylabel('$\gamma$ /cm$^{-1}$atm$^{-1}$ (line broadening)')
    #plt.ylabel('Spectral line intensity $S_(ij)$')
    plt.title('Air broadening of '+molecule+' given by HITRAN data, compared to that predicted by Buldyreva')
    plt.legend()
    plt.show()

#%%

#%%

#%%
