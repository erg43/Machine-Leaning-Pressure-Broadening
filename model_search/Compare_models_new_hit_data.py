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
import copy
#%%
# absolute path to folder containing data
rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/molecules_oracle/*'
#rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/molecules_oracle/*'

# be selective for data files
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-3:] == "csv" if "readme" not in f]

#file_list.insert(0, file_list.pop(49))
# read data files, taking the filename from absolute path
db = {}
for f in file_list:
    #i = f[94:-4]
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
        if 'gamma_air' in data.columns:
            # take needed parameters
            #if key[:4] == 'HONO':
            #    data = data[['J', 'Jpp', 'molecule_weight', 'm', 'findair', 'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox']]#, 'nu', 'sw', 'a']]
            data = data[['gamma_air', 'J', 'Jpp', 'molecule_weight', 'gamma_air-err', 'm', 'findair', 'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox']]#, 'nu', 'sw', 'a']]
        else:
            data = data[['J', 'Jpp', 'molecule_weight', 'm', 'findair', 'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox']]#, 'nu', 'sw', 'a']]

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
        if key == 'CH3CN':
            broadness_jeanna = broadening(data['m'][20], T, data['molecule_weight'][20], data['air_weight'][20], data['findair'][20])
        else:
            broadness_jeanna = broadening(data['m'][2], T, data['molecule_weight'][2], data['air_weight'][2], data['findair'][2])
        molecules[key]['broadness_jeanna'] = broadness_jeanna
        molecules[key] = molecules[key].drop(columns=['air_weight'])#symmetry

#%%
print(data.columns)
#%%
# weight data by error code, currently error code = weighting
for molecule, data in molecules.items():
    if molecule == 'HONO_prediction':
        continue
    # take weight as gamma-air-err
    weight = data['gamma_air-err']
    # Give helpful weightings
    # reweight 0 to tiny, because 0 gives /0 error
    weight2 = copy.deepcopy(weight)

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
#with open('search_ML_models_histgrad.pkl', 'rb') as f:
#    hist_model = pickle.load(f)

#with open('voter_model_best_histgrad.pkl', 'rb') as f:
#    hist_model = pickle.load(f)
#with open('search_ML_models_mlp_alp001_3tol.pkl', 'rb') as f:
#    mlp_model = pickle.load(f)
#with open('search_ML_models_mlp_best.pkl', 'rb') as f:
#    mlp_model = pickle.load(f)
with open('voter_model_final_09-23_training_new_HITRAN.pkl', 'rb') as f:
    new_model = pickle.load(f)
with open('voter_model_final_10-25_voter.pkl', 'rb') as f:
    new_model_again = pickle.load(f)
with open('voter_model_final_10-27_voter.pkl', 'rb') as f:
    new_model_polar = pickle.load(f)

#%%
#with open('search_ML_models_histgrad_data.pkl', 'rb') as f:
#    hist_data = pickle.load(f)
#with open('search_ML_models_mlp_alp001_3tol_data.pkl', 'rb') as f:
#    mlp_data = pickle.load(f)
with open('voter_model_final_09-23_training_new_HITRAN_data.pkl', 'rb') as f:
    new_data = pickle.load(f)
with open('voter_model_final_10-25_voter_data.pkl', 'rb') as f:
    new_data_again = pickle.load(f)
with open('voter_model_final_10-27_voter_data.pkl', 'rb') as f:
    new_data_polar = pickle.load(f)

#%%
'''i = 0
hist_models_list = []
for model in hist_model:
    labels = hist_data[i][0].split(',')
    print(labels)
    hist_models_list.append([labels, model])
    i+=1
   
i = 0
mlp_models_list = []
for model in mlp_model:
    labels = mlp_data[i][0].split(',')
    print(labels)
    mlp_models_list.append([labels, model[1]])
    i+=1
''' 
     
i = 0
new_models_list = []
for model in new_model:
    labels = new_data[i][0].split(',')
    print(labels)
    new_models_list.append([labels, model])
    i+=1
    
i = 0
new_models_list_again = []
for model in new_model_again:
    labels = new_data_again[i][0].split(',')
    print(labels)
    new_models_list_again.append([labels, model])
    i+=1
    
i = 0
new_models_list_polar = []
for model in new_model_polar:
    labels = new_data_polar[i][0].split(',')
    print(labels)
    new_models_list_polar.append([labels, model])
    i+=1

#%%
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#%%
cmap = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
cmap = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
#%%
available_styles = plt.style.available
#%%
print(available_styles)
#%%
import copy
from sklearn.metrics import mean_squared_error
from math import log10, floor

good = 0
bad = 0
no_hit = 0

scores_o = []
scores_n = []


mols_llist = []
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
import matplotlib as mpl

for molecule in molecules:
    #if molecule != 'NO2' and molecule != 'ClO':
    #    continue
    #if molecule != 'NO':
    #    continue
    #if molecule != ('CO'):
    #    continue
    data = molecules[molecule]
    for item in new_models_list_again:
        if molecule in item[0]:
            pipe_h = item[1]
            print(item)
    for item in new_models_list_polar:
        if molecule in item[0]:
            print(item)
            pipe_m = item[1]
    for item in new_models_list:
        if molecule in item[0]:
            #print(item)
            pipe_n = item[1]

    if molecule == 'HONO_prediction':
        continue
    #print(molecule)    
 
    data = data.sample(frac=1)
    
    data2 = data[data['errorbar'] != 0.5]
    data2 = data2[data2['errorbar'] != 0.6]
    data2 = data2[data2['errorbar'] != 0.7]
    data2 = data2[data2['errorbar'] != 0.8]
    if len(data2) != 0:

        X_test2 = data2.drop(['gamma_air', 'gamma_air-err', 'errorbar'], axis=1)[:1000]
        y_test2 = data2['gamma_air'][:1000]
        y_pred_n2 = pipe_n.predict(X_test2)
        mse_score_n2 = mean_squared_error(y_test2, y_pred_n2)
        
    

    
    #X_test = data[:1000]
    #print(X_test.columns)
    
    X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar'], axis=1)[:1000]
    y_test = data['gamma_air'][:1000]
    weight_test = data['gamma_air-err'][:1000]
    errorbar = np.mean(data['errorbar'])

    y_pred_h = pipe_h.predict(X_test)
    score_h = pipe_h.score(X_test, y_test, weight_test)
    y_pred_m = pipe_m.predict(X_test)
    score_m = pipe_m.score(X_test, y_test, weight_test)
    
    y_pred_n = pipe_n.predict(X_test)
    score_n = pipe_n.score(X_test, y_test, weight_test)
    
    #print(y_pred_n)
    #print(y_pred_m)
    #print(y_pred_h)
    
    #print('Scores:')
    #print('score hist = ')
    #print(score_o)
    #print('score new = ')
    #print(score_n)
 
    #mse_score_o = mean_squared_error(y_test, y_pred_o)
    mse_score_n = mean_squared_error(y_test, y_pred_n)
    
    #scores_o.append(score_o)
    scores_n.append(score_n)
    mols_llist.append(molecule)
 
    

    
    
    err_codes = data['errorbar'][:1000].value_counts().sort_index()
    data_by_vib_lev = {}
    #print('codes!')
    #print(err_codes)
    
    figure(figsize=((15, 7)), dpi=300)
    fig = plt.figure(1)
    plt.rc('font', size=18)
    
    
    toterr = 0
    i=0
    for code, gsfifbe in err_codes.iteritems():
        if molecule == 'C2H6':
            if code == 0.2:
                continue
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
        elif code==0.6:
            col=cmap[11]
        elif code==0.7:
            col=cmap[11]
        elif code==0.8:
            col=cmap[11]

        
        #print(code)
        if code in [0.5, 0.6, 0.7, 0.8]:
            label = 'HITRAN data, broadening values estimated from no empirical data'#, code = '+str(code)
            toterr += code*np.mean(data_level_y)*len(data_level_y)/1000
        else:
            #label = 'HITRAN data, error given by errorbar'
            label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), -int(floor(log10(abs(code*np.mean(data_level_y)))))))+f' cm$^{-1}$atm$^{-1}$'
            plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
            toterr += code*np.mean(data_level_y)*len(data_level_y)/1000
        #else:
        #    label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
        #    plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)


        
        plt.plot(data_level_x, data_level_y, 'x', label=label, color=col)
        
        
        i+=1

    
    plt.plot(X_test['M'], y_pred_h, '.', color=cmap[1], label="$\gamma$ predicted by retraining same model")
    plt.plot(X_test['M'], y_pred_m, '.', color=cmap[5], label="$\gamma$ predicted by changing polar")
    plt.plot(X_test['M'], y_pred_n, '.', color=cmap[3], label="$\gamma$ predicted by voting model")
    
    if len(data2) != 0:
        if len(err_codes.index) == 1:
            if err_codes.index != [0.5]:
                plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n2), color=cmap[3], label='RMSE of the ML Model')
            else:
                toterr = 0
            
        else:
            plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n2), color=cmap[3], label='RMSE of the ML Model')

    if len(err_codes.index) == 1 and err_codes.index == [0.5]:
        toterr = 0
    
    
    Air = 'Air'
    #plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
    plt.xlabel('m, rotational quantum number')
    plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
    plt.ylim(0.00, max([max(y_pred_n), max(y_test)])+0.01)
        
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2)
        
   
    #plt.legend()
    
    if toterr > np.sqrt(mse_score_n):
        good += 1
        print('GOOD!')
    elif toterr == 0:
        no_hit += 1
        print('NOHIT')
    else:
        bad += 1
        print('BAD')

    #if len(err_codes.index) == 1:
    #    if err_codes.index == [0.5]:
    #        plt.savefig('model_plot_for_paper/AAA_'+str(molecule))
    #    else:
    #        plt.savefig('model_plot_for_paper/AAA_'+str(molecule))
    #else:
    #    plt.savefig('model_plot_for_paper/AAA_'+str(molecule))
    plt.savefig('trial')
    plt.show()
    
    #plt.clf()
    break
#%%

totally = good + bad + no_hit
print(no_hit)
print(good)
print(bad)
print(totally)
print(good/totally)
print(bad/totally)
print(no_hit/totally)
print(good/(good+bad))
#%%
socres = {'old': scores_o, 'new': scores_n}
df = pd.DataFrame(socres)
df = df.set_index([pd.Index(mols_llist)])
df['percentage difference'] = 100* (df['new'] - df['old'])/(df['old']-1).abs()
df
#%%

#%%
print(df['percentage difference'].sum())
#%%
print('Scores:')
print('score old = ')
print(scores_o)
print('score new = ')
print(scores_n)
#%%

#%%

scores_o = []
scores_n = []

mols_llist = []
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
import matplotlib as mpl


data = molecules['HONO_prediction']
molecule = 'HONO'

X_test = data.sample(frac=1)[:1000]

y_pred_m = pipe_m.predict(X_test)

y_pred_n = pipe_n.predict(X_test)



figure(figsize=((15,7)), dpi=500)
fig = plt.figure(1)
plt.rc('font', size=18)

plt.plot(X_test['M'], y_pred_m, '.', color=cmap[5], label="$\gamma$ predicted by Voting Model")
#plt.plot(X_test['M'], y_pred_n, '.', color=cmap[1], label="$\gamma$ from new HITRAN data")

Air = 'Air'
#plt.title(
#    f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
plt.xlabel('M, rotational quantum number')
plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
plt.ylim(0, max([max(y_pred_n), max(y_pred_m)]) + 0.01)

#plt.legend()


if len(err_codes.index) == 1:
    if err_codes.index == [0.5]:
        plt.savefig('model_plot_for_paper/Paper_plot_' + str(molecule)+'_prediction.pdf')
    else:
        plt.savefig('model_plot_for_paper/Paper_plot_' + str(molecule)+'_prediction.pdf')
else:
    plt.savefig('model_plot_for_paper/Paper_plot_' + str(molecule)+'_prediction.pdf')
plt.show()
#plt.clf()


#%%
pipe_n.steps[1][1].named_estimators
#%%

from sklearn.inspection import permutation_importance

for name, submodel in pipe_n.steps[1][1].named_estimators.items():
    if name == 'hist':
        continue
    #if name == 'ada':
    #    continue
    if name == 'svr':
        continue
    if name == 'mlp':
        continue
    print(submodel)
    print(submodel.estimator_weights_)
    feature_importance = submodel.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(X_test.columns)[sorted_idx])
    plt.title("Feature Importance (MDI)")
    
    result = permutation_importance(
        submodel, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
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
y_pred_n = pipe_n.predict(X_test)
score_n = pipe_n.score(X_test, y_test, weight_test)
pipe_n
#%%
