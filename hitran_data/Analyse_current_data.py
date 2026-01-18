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



OCS
60.075
H2O2
34.0147
HF
20.00689
NH3
17.0305
H2
2.01588
ClO
50.96825
N2O
44.0128
H2CO
30.026
CO2
44.0095
H2O
18.0153
CH3F
34.0329
C2H2
26.0373
CO
28.0101
C2H4
28.0532
OH
17.0073
HCl
35.976793
C2H6
30.069
CH4
16.0425
NO
30.0061
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
print(len(db))
#%%
for key, data in db.items():
    if 'findair' in data.columns:
        print(key)
        if 'B0a' in data.columns:
            print(data.loc[0]['molecule_weight'])

#%%

    
molecules = {}

# take only molecules for which there is full data
for key, data in db.items():
    print(data.columns())
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
with open('search_ML_models_histgrad.pkl', 'rb') as f:
    hist_models = pickle.load(f)
with open('voter_model_best_voter.pkl', 'rb') as f:
    voter_models = pickle.load(f)
with open('voter_model_Where_does_voter_go_wrong.pkl', 'rb') as f:
    voter_models2 = pickle.load(f)
with open('search_ML_models_mlp_alp001_3tol.pkl', 'rb') as f:
    voter_models3 = pickle.load(f)
with open('search_ML_models_adaboost.pkl', 'rb') as f:
#with open('voter_model_try_mlp_to_scale.pkl', 'rb') as f:
    voter_models4 = pickle.load(f)
with open('search_ML_models_randomforest.pkl', 'rb') as f:
#with open('voter_model_try_mlp_to_scale_FULL_TRAINING.pkl', 'rb') as f:
    voter_models5 = pickle.load(f)
#%%

#with open("baseline_results.json", "r") as fp:
#    list_data = json.load(fp)

with open('search_ML_models_histgrad_data.pkl', 'rb') as f:
    list_data = pickle.load(f)
with open('voter_model_best_voter_data.pkl', 'rb') as f:
    list_data_compare = pickle.load(f)
with open('voter_model_Where_does_voter_go_wrong_data.pkl', 'rb') as f:
    voter_models2_d = pickle.load(f)
with open('search_ML_models_histgrad_data.pkl', 'rb') as f:
    voter_models3_d = pickle.load(f)
with open('search_ML_models_histgrad_data.pkl', 'rb') as f:
#with open('voter_model_try_mlp_to_scale_data.pkl', 'rb') as f:
    voter_models4_d = pickle.load(f)
with open('search_ML_models_randomforest_data.pkl', 'rb') as f:
#with open('voter_model_try_mlp_to_scale_FULL_TRAINING_data.pkl', 'rb') as f:
    voter_models5_d = pickle.load(f)

#%%
i = 0
named_hist_models = []
for model in hist_models:
    labels = list_data[i][0].split(',')
    named_hist_models.append([labels, model])
    i+=1
    
    
i = 0
named_voter_models = []
for model in voter_models:
    labels = list_data_compare[i][0].split(',')
    named_voter_models.append([labels, model])
    i+=1
    
i = 0
named_voter_models2 = []
for model in voter_models2:
    labels = voter_models2_d[i][0].split(',')
    named_voter_models2.append([labels, model[5]])
    i+=1
    
i = 0
named_voter_models3 = []
for model in voter_models3:
    labels = voter_models3_d[i][0].split(',')
    named_voter_models3.append([labels, model])
    i+=1
    
i = 0
named_voter_models4 = []
for model in voter_models4:
    labels = voter_models4_d[i][0].split(',')
    named_voter_models4.append([labels, model])#, model[1], model[2]])
    i+=1
    
i = 0
named_voter_models5 = []
for model in voter_models5:
    labels = voter_models5_d[i][0].split(',')
    named_voter_models5.append([labels, model])#, model[1], model[2]])
    i+=1
#%%
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#%%
cmap = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
cmap = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
#%%
import copy
from sklearn.metrics import mean_squared_error

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
    for item in named_voter_models2:
        if molecule in item[0]:
            pipe_v2 = item[1]
    for item in named_voter_models3:
        if molecule in item[0]:
            pipe_v3 = item[1]
    for item in named_voter_models4:
        if molecule in item[0]:
            pipe_v4 = item[1]
            #piper4 = item[2]
            #pipeer4 = item[3]
    for item in named_voter_models5:
        if molecule in item[0]:
            pipe_v5 = item[1]
            #piper5 = item[2]
            #pipeer5 = item[3]
    
                
    data = data.sample(frac=1)
    X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar'], axis=1)[:1000]
    y_test = data['gamma_air'][:1000]
    weight_test = data['gamma_air-err'][:1000]
    errorbar = np.mean(data['errorbar'])

    y_pred_h = pipe_h.predict(X_test)
    score_h = pipe_h.score(X_test, y_test, weight_test)

    y_pred_v = pipe_v.predict(X_test)
    score_v = pipe_v.score(X_test, y_test, weight_test)
    #X_test['y_test'] = y_test
    #X_test['y_pred'] = y_pred_v
    #X_test = X_test.sort_values(by=['M']).reset_index(drop=True)
    #for index, row, in X_test.iterrows():
    #    if index == len(X_test)-1:
    #        continue
    #    diff = X_test.iloc[index+1]['y_pred'] - row['y_pred']
    #    if row['M'] < 0:
    #        continue
    #    else:
    #        if diff > 0:
    #            X_test[index:]['y_pred'] = X_test[index:]['y_pred'] + diff
        
    
    mse_score = mean_squared_error(y_test, y_pred_v)
    
    y_pred_v2 = pipe_v2.predict(X_test)
    score_v2 = pipe_v2.score(X_test, y_test, weight_test)
    y_pred_v3 = pipe_v3.predict(X_test)
    score_v3 = pipe_v3.score(X_test, y_test, weight_test)
    y_pred_v4 = pipe_v4.predict(X_test)
    score_v4 = pipe_v4.score(X_test, y_test, weight_test)
    y_pred_v5 = pipe_v5.predict(X_test)
    score_v5 = pipe_v5.score(X_test, y_test, weight_test)
    
    scores_h.append(score_h)
    scores_v.append(score_v)
    mols_llist.append(molecule)
    
    '''
    
    cols_vals_test = []
    mean_y_test = []
    mean_y_pred_v4_1 = []
    mean_y_pred_v5_1 = []
    for weight in X_test['molecule_weight'].unique():
        myp4 = np.mean(y_pred_v4[X_test['molecule_weight']==weight])
        mean_y_pred_v4_1.append(myp4)
        myp5 = np.mean(y_pred_v5[X_test['molecule_weight']==weight])
        mean_y_pred_v5_1.append(myp5)
        mean_y_test.append(np.mean(y_test[X_test['molecule_weight'] == weight]))
        cols_vals_test.append(X_test[X_test['molecule_weight']==weight].drop(['J', 'Jpp', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox', 'M'], axis=1).iloc[0])
    cols_vals_test = pd.concat(cols_vals_test, axis=1).T
    
    mean_y_pred_v4_2 = piper4.predict(cols_vals_test)
    mean_y_pred_v5_2 = piper5.predict(cols_vals_test)    
    
    y_pred_v4_3 = pipeer4.predict(X_test)
    y_pred_v5_3 = pipeer5.predict(X_test)

    y_pred_final_v4_2 = copy.copy(y_pred_v4)
    y_pred_final_v4_3 = copy.copy(y_pred_v4)
    y_pred_final_v5_2 = copy.copy(y_pred_v5)
    y_pred_final_v5_3 = copy.copy(y_pred_v5)
    mean_y_test_v4_3 = []
    mean_y_test_v5_3 = []
    n=0
    for weight in X_test['molecule_weight'].unique():
        mean_y_pred_v4_3 = np.mean(y_pred_v4_3[X_test['molecule_weight'] == weight])
        y_pred_final_v4_3[X_test['molecule_weight']==weight] = mean_y_pred_v4_3 / mean_y_pred_v4_1[n] * y_pred_v4[X_test['molecule_weight']==weight]
        y_pred_final_v4_2[X_test['molecule_weight']==weight] = mean_y_pred_v4_2[n] / mean_y_pred_v4_1[n] * y_pred_v4[X_test['molecule_weight']==weight]
        mean_y_pred_v5_3 = np.mean(y_pred_v5_3[X_test['molecule_weight'] == weight])
        y_pred_final_v5_3[X_test['molecule_weight']==weight] = mean_y_pred_v5_3 / mean_y_pred_v5_1[n] * y_pred_v5[X_test['molecule_weight']==weight]
        y_pred_final_v5_2[X_test['molecule_weight']==weight] = mean_y_pred_v5_2[n] / mean_y_pred_v5_1[n] * y_pred_v5[X_test['molecule_weight']==weight]
        n+=1    
    
    
    
    '''
    
    print(molecule)
    #print(score_h, score_v)
    #print(np.mean(weight_test))
    
    
    err_codes = data['errorbar'][:1000].value_counts().sort_index()
    data_by_vib_lev = {}
    print(err_codes.index)
    
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    
    
    #plt.plot(X_test['M'], y_pred_h, '.', color=cmap[0], label="histgrad model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v, '.', color=cmap[1], label="voter model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v2, '.', color=cmap[2], label="voter no mlp Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v3, '.', color=cmap[3], label="mlp model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v4, '.', color=cmap[8], label="adaboost model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v5, '.', color=cmap[9], label="random forest model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v4, 'ko', label="small 1 model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v5, 'yo', label="big 1 model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_final_v4_2, 'go', label="small 2 model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_final_v5_2, 'ro', label="big 2 model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_final_v4_3, 'co', label="small 3 model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_final_v5_3, 'mo', label="big_3 model Predicted $\gamma$")
    print(err_codes)
    
    #c = np.array([20, 5, 2, 1, .7, .5])
    
    #norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    #cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    #cmap.set_array([])

    i=0
    for code, gsfifbe in err_codes.iteritems():
        #for code in err_codes.index:
        data_level_x = X_test['M'][data['errorbar']==code]
        data_level_y = y_test[data['errorbar']==code]
        
                
        if code==0.01:
            col=cmap[10]
        elif code==0.02:
            col=cmap[6]
        elif code==0.05:
            col=cmap[7]
        elif code==0.1:
            col=cmap[4]
        elif code==0.2:
            col=cmap[5]
        elif code==0.5:
            col=cmap[11]
        
        if len(err_codes.index) == 1:
            if err_codes.index == [0.5]:
                label = 'HITRAN data, broadening values estimated from no empirical data'
            else:
                label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
                plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
        else:
            label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
            plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)


        
        plt.plot(data_level_x, data_level_y, 'x', label=label, color=col)
        
        i+=1



    plt.plot(X_test['M'], y_pred_v2, '.', color=cmap[2], label="voter model with mlp excluded, Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_v4, '.', color=cmap[0], label="adaboost model Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_v5, '.', color=cmap[8], label="random forest model Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_v3, '.', color=cmap[9], label="mlp model Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_h, '.', color=cmap[1], label="gradient boosting model Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_v, '.', color=cmap[3], label="voter model Predicted $\gamma$")
    #if len(err_codes.index) == 1:
    #    if err_codes.index != [0.5]:
    #        plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score), color=cmap[3])
    #else:
    #    plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score), color=cmap[3])

    
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
    plt.ylim(0)
    #print(errorbar)
    

    #plt.ylim(0)
    plt.legend()
    if len(err_codes.index) == 1:
        if err_codes.index == [0.5]:
            #plt.ylim(0, 0.12)
            plt.savefig('model_plot_for_paper/model_comparison_plot_'+str(molecule))
        else:
            plt.savefig('model_plot_for_paper/model_comparison_plot_'+str(molecule))
    else:
        plt.savefig('model_plot_for_paper/model_comparison_plot_'+str(molecule))
    plt.show()

#%%
scores_h = []
scores_v = []
mols_llist = []
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']

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
    
    
    err_codes = data['errorbar'].value_counts().sort_index()
    data_by_vib_lev = {}
    print(err_codes.index)
    
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    
    
    plt.plot(X_test['M'], y_pred_h, 'o', label="hist Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_v, 'o', label="voter Predicted $\gamma$")
    
    i=0
    for code, gsfifbe in err_codes.iteritems():
        print(code)
        if code == 0.5:
            continue
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
    for item in named_voter_models2:
        if molecule in item[0]:
            pipe_v2 = item[1]
    for item in named_voter_models3:
        if molecule in item[0]:
            pipe_v3 = item[1]
    for item in named_voter_models4:
        if molecule in item[0]:
            pipe_v4 = item[1]
    for item in named_voter_models5:
        if molecule in item[0]:
            pipe_v5 = item[1]

            
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
socres = {'histgrad': scores_h, 'voting': scores_v}
df = pd.DataFrame(socres)
df = df.set_index([pd.Index(mols_llist)])
df['percentage difference'] = 100* (df['voting'] - df['histgrad'])/(df['histgrad']-1).abs()
df
#%%
print(df['percentage difference'].sum())
#%% md
# 