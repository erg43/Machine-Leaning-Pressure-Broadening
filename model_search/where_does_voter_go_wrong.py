#%%
masses = {'26.0373': 'C2H2',
            '28.0532': 'C2H4',
            '30.069000000000006': 'C2H6',
            '32.0419': 'CH3OH',
            '34.0329': 'CH3F',
            '16.0425': 'CH4',
            '28.0101': 'CO',
            '44.0095': 'CO2',
            '50.96825': 'ClO',
            '52.9653': 'ClO',
            '66.0069': 'COF2',
            '2.01588': 'H2',
            '18.0153': 'H2O',
            '30.026': 'H2CO',
            '34.0147': 'H2O2',
            '44.07100000000001': 'CS',
            '20.00689': 'HF',
            '28.0134': 'N2',
            '33.0067': 'HO2',
            '35.976793': 'HCl',
            '37.973843': 'HCl',
            '79.926277': 'HBr',
            '81.924231': 'HBr',
            '17.0305': 'NH3',
            '30.0061': 'NO',
            '31.9988': 'O2',
            '44.0128': 'N2O',
            '46.0055': 'NO2',
            '17.0073': 'OH',
            '47.9982': 'O3',
            '60.075': 'OCS',
            '64.064': 'SO2'}
#%%
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
#from sklearn.ensemble import HistGradientBoostingRegressor
from matplotlib.pyplot import figure
from sklearn.neural_network import MLPRegressor
import pickle
#with open("baseline_results.json", "r") as fp:
#    list_data = json.load(fp)
#%%
# absolute path to folder containing data
rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/molecules_oracle/*'
# be selective for data files
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-3:] == "csv" if "readme" not in f]

# read data files, taking the filename from absolute path
db = {}
for f in file_list:
    i = f[94:-4]
    print(i)
    db[i] = pd.read_csv(f)
#%%
# Set parameters
T = 298      # Kelvin


# define Jeanna's broadening formalism for use later.  Taken from paper ...
def broadening(m, T, ma, mp, b0):
    gamma = 1.7796e-5 * (m/(m-2)) * (1/np.sqrt(T)) * np.sqrt((ma+mp)/(ma*mp)) * b0**2
    return(gamma)



#%%

    
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
        data = data.drop(columns=['P', 'Q', 'R', 'O', 'S'])
        # assign data to molecule
        molecules[key] = data
        

        # calculate jeanna broadness, and add to dictionary
        broadness_jeanna = broadening(data['m'][2], T, data['molecule_weight'][2], data['air_weight'][2], data['findair'][2])
        molecules[key]['broadness_jeanna'] = broadness_jeanna
        molecules[key] = molecules[key].drop(columns=['air_weight'])#symmetry

#%%
# weight data by error code, currently error code = weighting
for molecule, data in molecules.items():
    # take weight as gamma-air-err
    weight = data['gamma_air-err']
    # Give helpful weightings
    # reweight 0 to tiny, because 0 gives /0 error
    weight2 = weight

    weight2 = weight2.replace(0, (.5))    # 0  ~~~  unreported or unavailable
    weight2 = weight2.replace(1, (.5))    # 1  ~~~  Default or constant
    weight2 = weight2.replace(2, (.5))    # 2  ~~~  Average or estimate
    weight2 = weight2.replace(3, (.5))     # 3  ~~~  err >= 20 %              50
    weight2 = weight2.replace(4, (.2))     # 4  ~~~  20 >= err >= 10 %        15
    weight2 = weight2.replace(5, (.1))    # 5  ~~~  10 >= err >= 5 %         7.5
    weight2 = weight2.replace(6, (.05))    # 6  ~~~  5 >= err >= 2 %          3.5
    weight2 = weight2.replace(7, (.02))    # 7  ~~~  2 >= err >= 1 %          1.5
    weight2 = weight2.replace(8, (.01))    # 8  ~~~  err <= 1 %               0.5
    
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
    molecules[molecule]['errorbar'] = weight2
#%%
#with open('voter_model_best_histgrad.pkl', 'rb') as f:
with open('voter_model_Where_does_voter_go_wrong.pkl', 'rb') as f:
    voter_models = pickle.load(f)
#with open('search_ML_models_mlp_alp001_3tol_data.pkl', 'rb') as f:
#with open('voter_model_best_voter.pkl', 'rb') as f:
#with open('search_ML_models_histgrad.pkl', 'rb') as f:
#    voter_models = pickle.load(f)
#%%


#with open('voter_model_best_histgrad_data.pkl', 'rb') as f:
with open('voter_model_Where_does_voter_go_wrong_data.pkl', 'rb') as f:
    list_data = pickle.load(f)
#with open('search_ML_models_mlp_alp001_3tol_data.pkl', 'rb') as f:
#with open('voter_model_best_voter_data.pkl', 'rb') as f:
#with open('search_ML_models_histgrad_data.pkl', 'rb') as f:
#    list_data_compare = pickle.load(f)

#%%
i = 0
named_hist_models = []
for models in voter_models:
    model_full = models[0]
    model_min1 = models[1]
    model_min2 = models[2]
    model_min3 = models[3]                
    model_min4 = models[4]
    model_min5 = models[5]
    labels = list_data[i][0].split(',')
    print(labels)
    named_hist_models.append([labels, [model_full, model_min1, model_min2, model_min3, model_min4, model_min5]])
    i+=1
    
    
#i = 0
#named_voter_models = []
#for model in voter_models:
#    labels = list_data_compare[i][0].split(',')
#    named_voter_models.append([labels, model])
#    i+=1
#%%
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#%%
socres = {'histgrad': scores_h, 'voting': scores_v}
df = pd.DataFrame(socres)
df = df.set_index([pd.Index(mols_llist)])
df['percentage difference'] = 100* (df['voting'] - df['histgrad'])/(df['histgrad']-1).abs()
df
#%%
print(df['percentage difference'].sum())
#%%
for molecule in molecules:
    data = molecules[molecule]
    print(data['molecule_weight'].unique())
#%%
scores_full = []
scores_min1 = []
scores_min2 = []
scores_min3 = []
scores_min4 = []
scores_min5 = []
mols_llist = []
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']

for molecule in molecules:
    data = molecules[molecule]
    for item in named_hist_models:
        if molecule in item[0]:
            pipe_hs = item[1]
            pipe_full = pipe_hs[0]
            pipe_min1 = pipe_hs[1]
            pipe_min2 = pipe_hs[2]
            pipe_min3 = pipe_hs[3]                
            pipe_min4 = pipe_hs[4]
            pipe_min5 = pipe_hs[5]
    
    
    data = data.sample(frac=1)
    X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar'], axis=1)[:1000]
    y_test = data['gamma_air'][:1000]
    weight_test = data['gamma_air-err'][:1000]
    errorbar = np.mean(data['errorbar'])

    y_pred_full = pipe_full.predict(X_test)
    score_full = pipe_full.score(X_test, y_test, weight_test)
    y_pred_min1 = pipe_min1.predict(X_test)
    score_min1 = pipe_min1.score(X_test, y_test, weight_test)
    y_pred_min2 = pipe_min2.predict(X_test)
    score_min2 = pipe_min2.score(X_test, y_test, weight_test)
    y_pred_min3 = pipe_min3.predict(X_test)
    score_min3 = pipe_min3.score(X_test, y_test, weight_test)
    y_pred_min4 = pipe_min4.predict(X_test)
    score_min4 = pipe_min4.score(X_test, y_test, weight_test)
    y_pred_min5 = pipe_min5.predict(X_test)
    score_min5 = pipe_min5.score(X_test, y_test, weight_test)

    scores_full.append(score_full)
    scores_min1.append(score_min1)
    scores_min2.append(score_min2)
    scores_min3.append(score_min3)                
    scores_min4.append(score_min4)
    scores_min5.append(score_min5)

    mols_llist.append(molecule)    
    print(molecule)
    #print(score_h, score_v)
    #print(np.mean(weight_test))
    
    
    err_codes = data['errorbar'].value_counts().sort_index()
    data_by_vib_lev = {}
    print(err_codes.index)
    
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    
    
    plt.plot(X_test['M'], y_pred_full, 'o', label="full voter Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_min1, 'o', label="voter min 1 Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_min2, 'o', label="voter min 2 Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_min3, 'o', label="voter min 3 Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_min4, 'o', label="voter min 4 Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_min5, 'o', label="voter min 5 Predicted $\gamma$")
    
    i=0
    for code, gsfifbe in err_codes.iteritems():
        print(code)
        #if code == 0.5:
        #    continue
        #for code in err_codes.index:
        data_level_x = X_test['M'][data['errorbar']==code]
        data_level_y = y_test[data['errorbar']==code]
        label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
        plt.plot(data_level_x, data_level_y, 'x', label=label, color=colours[i])
        plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=colours[i])
        i+=1


    #plt.plot(X_test['M'], y_test, 'x', label='HITRAN $\gamma$ data')

    #plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    #plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
    #plt.plot(x_plot[:1000], y_pred2[:1000], 'o', label="Predicted $\gamma$ - 1ov")
    #plt.plot(x_plot[:1000], y_pred3[:1000], 'o', label="Predicted $\gamma$ - err")
    #plt.plot(x_plot[:1000], y_pred4[:1000], 'o', label="Predicted $\gamma$ - sqerr")
    #plt.plot(x_plot[:1000], y_pred5[:1000], 'o', label="Predicted $\gamma$ - 1ovsqrt")
    #plt.plot(x_plot[:1000], y_pred6[:1000], 'o', label="Predicted $\gamma$ - noweight")
    #plt.plot(x_plot[:1000], y_pred7[:1000], 'o', label="Predicted $\gamma$ - mlpreg")
    Air = 'Air'
    #plt.axhline(y=X_test.iloc[0]['broadness_jeanna'], linestyle='-', label='Jeanna broadening for 298K')
    plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
    plt.xlabel('M, rotational quantum number')
    plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
    #plt.annotate('weighting = '+str(np.mean(weight_test)), (25,0.1))
    #plt.ylim(0)
    #print(errorbar)
    

    #plt.ylim(0)
    plt.legend()
    
    plt.show()
#%%
total_errs = 0
total_data = 0


for molecule in molecules:
    print(molecule)
    data = molecules[molecule]
    for item in named_hist_models:
        if molecule in item[0]:
            pipe_h = item[1]
    for item in named_voter_models:
        if molecule in item[0]:
            pipe_v = item[1]
    X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar'], axis=1)
    y_test = data['gamma_air']
    weight_test = data['gamma_air-err']
    errors = data['errorbar']*y_test
    errorbar = np.mean(data['errorbar'])

    y_pred_h = pipe_h.predict(X_test)
    y_pred_v = pipe_v.predict(X_test)
    
    total_errs += (abs(y_pred_v - y_test)/errors>1).sum()
    total_data += len(y_test)
    #print((abs(y_pred_h - y_test)/errors>1).sum())
    print(np.mean(abs(y_pred_h - y_test)/errors))

print(total_errs)
print(total_errs/total_data)


#%%
scores_h = []
scores_v = []
mols_llist = []
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
import matplotlib as mpl

for molecule in molecules:
    data = molecules[molecule]
    for item in named_hist_models:
        if molecule in item[0]:
            pipe_h = item[1]
    for item in named_voter_models:
        if molecule in item[0]:
            pipe_v = item[1]
    
    data = data.sample(frac=1)
    X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar'], axis=1)[:1000]
    y_test = data['gamma_air'][:1000]
    weight_test = data['gamma_air-err'][:1000]
    errorbar = np.mean(data['errorbar'])

    y_pred_h = pipe_h.predict(X_test)
    score_h = pipe_h.score(X_test, y_test, weight_test)

    y_pred_v = pipe_v.predict(X_test)
    score_v = pipe_v.score(X_test, y_test, weight_test)
    
    scores_h.append(score_h)
    scores_v.append(score_v)
    mols_llist.append(molecule)
    
    
    print(molecule)
    #print(score_h, score_v)
    #print(np.mean(weight_test))
    
    
    err_codes = data['errorbar'][:1000].value_counts().sort_index()
    data_by_vib_lev = {}
    print(err_codes.index)
    
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    
    
    plt.plot(X_test['M'], y_pred_h, 'go', label="hist Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_v, 'ro', label="voter Predicted $\gamma$")
    
    
    c = np.array([6, 5, 4, 3, 2, 1])
    
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])

    i=5
    for code, gsfifbe in err_codes.iteritems():
        #for code in err_codes.index:
        data_level_x = X_test['M'][data['errorbar']==code]
        data_level_y = y_test[data['errorbar']==code]
        label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
        
        
        plt.plot(data_level_x, data_level_y, 'x', label=label, color=cmap.to_rgba(i))
        plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=cmap.to_rgba(i))
        i-=1


    #plt.plot(X_test['M'], y_test, 'x', label='HITRAN $\gamma$ data')

    #plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    #plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
    #plt.plot(x_plot[:1000], y_pred2[:1000], 'o', label="Predicted $\gamma$ - 1ov")
    #plt.plot(x_plot[:1000], y_pred3[:1000], 'o', label="Predicted $\gamma$ - err")
    #plt.plot(x_plot[:1000], y_pred4[:1000], 'o', label="Predicted $\gamma$ - sqerr")
    #plt.plot(x_plot[:1000], y_pred5[:1000], 'o', label="Predicted $\gamma$ - 1ovsqrt")
    #plt.plot(x_plot[:1000], y_pred6[:1000], 'o', label="Predicted $\gamma$ - noweight")
    #plt.plot(x_plot[:1000], y_pred7[:1000], 'o', label="Predicted $\gamma$ - mlpreg")
    Air = 'Air'
    #plt.axhline(y=X_test.iloc[0]['broadness_jeanna'], linestyle='-', label='Jeanna broadening for 298K')
    plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
    plt.xlabel('M, rotational quantum number')
    plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
    #plt.annotate('weighting = '+str(np.mean(weight_test)), (25,0.1))
    #plt.ylim(0)
    #print(errorbar)
    

    #plt.ylim(0)
    plt.legend()
    
    plt.show()
#%%
