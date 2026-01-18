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


file_list.insert(0, file_list.pop(49))

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
# take only molecules for which there is full data
for key, data in db.items():
    print(key)
    if key == 'SO3':
        continue
    print(data['m'][0])
    print(data['molecule_dipole'][0])
    print()
#%%

    
molecules = {}

# take only molecules for which there is full data
for key, data in db.items():
    # filter for rotational constant 3D - as that was the last thing added
    if 'B0a' in data.columns:
        # take needed parameters
        if key[:4] == 'HONO':
            data = data[['J', 'Jpp', 'molecule_weight', 'm', 'findair', 'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox']]#, 'nu', 'sw', 'a']]
        else:
            data = data[['gamma_air', 'J', 'Jpp', 'molecule_weight', 'gamma_air-err', 'm', 'findair', 'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox']]#, 'nu', 'sw', 'a']]

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
for molecule, data in molecules.items():
    if molecule != 'OH':
        continue
    #print(molecule)
    bb = data['gamma_air-err'].value_counts()
    if len(bb)==2:
        print(molecule)
        print(bb.index)
        print(bb)
        print()
        print()
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

#%%
# absolute path to folder containing data
rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/*'
rootdir_glob2 = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/plots for paper/*'
rootdir_glob3 = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/plotting_oracle/*'
rootdir_glob4 = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/compare_model_types/*'


# be selective for data files
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-3:] == "pkl"]

file_list2 = [f for f in iglob(rootdir_glob2, recursive=True) if os.path.isfile(f) if f[-3:] == "pkl"]
file_list3 = [f for f in iglob(rootdir_glob3, recursive=True) if os.path.isfile(f) if f[-3:] == "pkl"]
file_list4 = [f for f in iglob(rootdir_glob4, recursive=True) if os.path.isfile(f) if f[-3:] == "pkl"]

#%%

# read data files, taking the filename from absolute path
runs = {}
for f in file_list:
    #i = f[94:-4]
    i = f[77:-4]
    print(i)
    with open(f, 'rb') as file:
        datash = pickle.load(file)
    runs[i] = datash

for f in file_list2:
    #i = f[94:-4]
    i = f[80:-4]
    print(i)
    with open(f, 'rb') as file:
        datash = pickle.load(file)
    runs[i] = datash

for f in file_list3:
    #i = f[94:-4]
    i = f[80:-4]
    print(i)
    with open(f, 'rb') as file:
        datash = pickle.load(file)
    runs[i] = datash
    
for f in file_list4:
    #i = f[94:-4]
    i = f[97:-4]
    if i[-6:] == 'linear':
        continue
    print(i)
    with open(f, 'rb') as file:
        datash = pickle.load(file)
    runs[i] = datash
#%%
print(len(runs))
keys = runs.keys()
for key in keys:
    if key[:6] == 'search':
        print(key)
for key in keys:
    if key[:16] == 'voter_model_hist':
        print(key)
for key in keys:
    if key[:17] == 'voter_model_voter':
        print(key)
    #else:
    #    print(key)


#%%

for key, data in runs.items():
    if key[:6] == 'search' or key[:16] == 'voter_model_hist' or key[:17] == 'voter_model_voter':
        if key[-4:] != 'data':
            models_list = []
            print(key)
            i=0
            for model in data:
                labels = runs['search_ML_models_histgrad_data'][i][0].split(',')
                models_list.append([labels, model])
                i+=1
            print(models_list)
            runs[key] = models_list
        
            

#%%
for key, data in runs.items():
    if key[:20] == 'search_ML_models_mlp':
        if key[-4:] == 'ters':
            continue
        if key[-4:] != 'data':
            print(key)
            print(data)
            
            break
#%%
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#%%
cmap = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
cmap = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
#%%
for key, datsh in runs.items():
    print(key)
#%%
import copy
from sklearn.metrics import mean_squared_error
from math import log10, floor

good = 0
bad = 0
no_hit = 0


scores_o = []
scores_n = []
mses = []
kikis = []
mols_llist = []
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
import matplotlib as mpl


for key, datsh in runs.items():
    if key[:6] == 'search' or key[:16] == 'voter_model_hist' or key[:17] == 'voter_model_voter':
        if key[-4:] != 'data':
            if key[-10:] == 'parameters':
                continue
            pipe_n = datsh[0][1]
            scores_n_key = []
            mols_llist_key = []
            mses_key = []
            minsss = []
            for molecule in molecules:
                print(molecule)
                #if molecule == 'CS2':
                #    continue
                #if molecule in ['GeH4', 'CS', 'CS2', 'O3', 'HBr', 'O2', 'PH3', 'N2', 'HO2', 'SO2', 'COCl2', 'HC3N', 'CH3OH', 'OCS', 'HOBr', 'CH3I', 'H2O2', 'H2S', 'CH3Cl', 'HF', 'NH3', 'H2', 'CH3CN', 'COF2']:
                #    continue
                data = molecules[molecule]
                #print(molecule)
                
                for item in datsh:
                    if molecule in item[0]:
                        pipe_n = item[1]
                if molecule == 'HONO_prediction':
                    continue
                #print(molecule) 
                #print(key)
             
                data = data.sample(frac=1)
                #data = data[data['errorbar'] != 0.5]
                #data = data[data['errorbar'] != 0.6]
                #data = data[data['errorbar'] != 0.7]
                #data = data[data['errorbar'] != 0.8]
                #if len(data) == 0:
                #    continue
                X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar'], axis=1)[:1000]
                y_test = data['gamma_air'][:1000]
                weight_test = data['gamma_air-err'][:1000]
                errorbar = np.mean(data['errorbar'])
            
                #y_pred_o = pipe_o.predict(X_test)
                #score_o = pipe_o.score(X_test, y_test, weight_test)
            
                y_pred_n = pipe_n.predict(X_test)
                score_n = pipe_n.score(X_test, y_test, weight_test)
                vee = ((y_test- y_test.mean()) ** 2).sum()
                
                score_n2 = score_n*vee
                print(molecule)
                print('Scores:')
                print('score = ')
                print(score_n)
                print('this score should be worse')
                print(score_n2)
             
                #mse_score_o = mean_squared_error(y_test, y_pred_o)
                mse_score_n = 100*np.sqrt(mean_squared_error(y_test, y_pred_n))/np.mean(y_test)
                print('mse = %')
                print(mse_score_n)
                
                print(np.mean(weight_test))
                print()
                #scores_o.append(score_o)
                scores_n_key.append(score_n2)
                mols_llist_key.append(molecule)
                mses_key.append(mse_score_n)
                
                mean_errorbars = np.mean(data['errorbar']*np.mean(y_test))
                minsss.append(100*mean_errorbars/np.mean(y_test))
                '''
                err_codes = data['errorbar'][:1000].value_counts().sort_index()
                data_by_vib_lev = {}
                #print('codes!')
                #print(err_codes)
                
                figure(figsize=((15, 7)), dpi=500)
                fig = plt.figure(1)
                plt.rc('font', size=14)
                
               
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
                        label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), -int(floor(log10(abs(code*np.mean(data_level_y)))))))+f' cm$^{-1}$atm$^{-1}$'
                        plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
                        toterr += code*np.mean(data_level_y)*len(data_level_y)/1000
                    #else:
                    #    label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
                    #    plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
            
            
                    
                    plt.plot(data_level_x, data_level_y, 'x', label=label, color=col)
                    
                    
                    i+=1
            
            
                #plt.plot(X_test['M'], y_pred_o, '.', color=cmap[3], label="$\gamma$ from older HITRAN data")
                plt.plot(X_test['M'], y_pred_n, '.', color=cmap[1], label="$\gamma$ from new HITRAN data")
            
                
                if len(err_codes.index) == 1:
                    if err_codes.index != [0.5]:
                        plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n), color=cmap[3], label='RMSE of the Voter Model')
                    else:
                        toterr = 0
                        
                else:
                    plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n), color=cmap[3], label='RMSE of the Voter Model')
            
            
                Air = 'Air'
                plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
                plt.xlabel('M, rotational quantum number')
                plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
                plt.ylim(0, max([max(y_pred_n), max(y_test)])+0.01)
            
                plt.legend()
                
                if toterr > np.sqrt(mse_score_n):
                    good += 1
                    print('GOOD!')
                elif toterr == 0:
                    no_hit += 1
                    print('NOHIT')
                else:
                    bad += 1
                    print('BAD')
                    
                if len(err_codes.index) == 1:
                    if err_codes.index == [0.5]:
                        plt.savefig('model_plot_for_paper/oldnew_comparison_plot_'+str(molecule))
                    else:
                        plt.savefig('model_plot_for_paper/oldnew_comparison_plot_'+str(molecule))
                else:
                    plt.savefig('model_plot_for_paper/oldnew_comparison_plot_'+str(molecule))
                plt.show()
                plt.clf()
                '''
                

                
                
                
                
            mols_llist.append(mols_llist_key)
            scores_n.append(scores_n_key)
            kikis.append(key)
            mses.append(mses_key)
#%%


d = {'molecule': mols_llist[3], 'score': scores_n[3], 'mse': mses[3]}
df = pd.DataFrame(data=d)
#%%
df
#%%
print(scores_n)
#%%
print(mols_llist)
#%%
print(kikis)
#%%
print(mses)
#%%
print(minsss)
#%%
print(np.mean(minsss))
#%%
mseeeeees = []
for msees in mses:
    mse = np.mean(msees)
    mseeeeees.append((mse))
d = {'model': kikis, 'average mse': mseeeeees}
df = pd.DataFrame(data=d)
#%%
print(df)
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

y_pred_o = pipe_o.predict(X_test)

y_pred_n = pipe_n.predict(X_test)



figure(figsize=((15, 7)), dpi=500)
fig = plt.figure(1)
plt.rc('font', size=14)

plt.plot(X_test['M'], y_pred_o, '.', color=cmap[3], label="$\gamma$ from older HITRAN data")
#plt.plot(X_test['M'], y_pred_n, '.', color=cmap[1], label="$\gamma$ from new HITRAN data")

Air = 'Air'
plt.title(
    f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
plt.xlabel('M, rotational quantum number')
plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
plt.ylim(0, max([max(y_pred_n), max(y_pred_o)]) + 0.01)

plt.legend()


#if len(err_codes.index) == 1:
#    if err_codes.index == [0.5]:
#        plt.savefig('model_plot_for_paper/oldnew_comparison_plot_' + str(molecule))
#    else:
#        plt.savefig('model_plot_for_paper/oldnew_comparison_plot_' + str(molecule))
#else:
#    plt.savefig('model_plot_for_paper/oldnew_comparison_plot_' + str(molecule))
plt.show()
#plt.clf()


#%%
print(np.shape(runs['voter_model_Where_does_voter_go_wrong']))
print(np.shape(runs['voter_model_Where_does_voter_go_wrong_data']))
#%%
models = runs['voter_model_Where_does_voter_go_wrong']
print(models[1][1][1])
#%%

models = runs['voter_model_Where_does_voter_go_wrong']
models_list = []
print(len(models))
i=0
for model in models:
    model_full = model[0]
    model_min1 = model[1]
    model_min2 = model[2]
    model_min3 = model[3]             
    model_min4 = model[4]
    model_min5 = model[5]
    labels = runs['voter_model_Where_does_voter_go_wrong_data'][i][0].split(',')
    print(labels)
    #for mode in model:
        
    models_list.append([labels, [model_full, model_min1, model_min2, model_min3, model_min4, model_min5]])
    i+=1
print(len(models_list))
runs['voter_model_Where_does_voter_go_wrong'] = models_list
#%%
print(np.mean(mses_full))
print(np.mean(mses_min1))
print(np.mean(mses_min2))
print(np.mean(mses_min3))
print(np.mean(mses_min4))
print(np.mean(mses_min5))
import copy
from sklearn.metrics import mean_squared_error
from math import log10, floor

good = 0
bad = 0
no_hit = 0


mses_full = []
mses_min1 = []
mses_min2 = []
mses_min3 = []
mses_min4 = []
mses_min5 = []

scoreeo_full = []
scoreeo_min1 = []
scoreeo_min2 = []
scoreeo_min3 = []
scoreeo_min4 = []
scoreeo_min5 = []



scores_o = []
scores_n = []
mses = []
kikis = []
mols_llist = []
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
import matplotlib as mpl

for key, datsh in runs.items():
    if key == 'voter_model_Where_does_voter_go_wrong':
        pipe_n = datsh[0][1]
        scores_n_key = []
        mols_llist_key = []
        mses_key = []
        minsss = []
        mses_full_key = []
        mses_min1_key = []
        mses_min2_key = []
        mses_min3_key = []
        mses_min4_key = []
        mses_min5_key = []
        
        scores_full_key = []
        scores_min1_key = []
        scores_min2_key = []
        scores_min3_key = []
        scores_min4_key = []
        scores_min5_key = []

        for molecule in molecules:
            #if molecule != 'NO':
            #    continue
            #if molecule in ['GeH4', 'CS', 'CS2', 'O3', 'HBr', 'O2', 'PH3', 'N2', 'HO2', 'SO2', 'COCl2', 'HC3N', 'CH3OH', 'OCS', 'HOBr', 'CH3I', 'H2O2', 'H2S', 'CH3Cl', 'HF', 'NH3', 'H2', 'CH3CN', 'COF2']:
            #    continue
            data = molecules[molecule]
            #print(molecule)
    
            for item in datsh:
                if molecule in item[0]:
                    
                    pipe_hs = item[1]
                    print(pipe_hs)
                    pipe_full = pipe_hs[0]
                    pipe_min1 = pipe_hs[1]
                    pipe_min2 = pipe_hs[2]
                    pipe_min3 = pipe_hs[3]                
                    pipe_min4 = pipe_hs[4]
                    pipe_min5 = pipe_hs[5]
            if molecule == 'HONO_prediction':
                continue
            #print(molecule) 
            #print(key)
    
            data = data.sample(frac=1)
            data = data[data['errorbar'] != 0.5]
            data = data[data['errorbar'] != 0.6]
            data = data[data['errorbar'] != 0.7]
            data = data[data['errorbar'] != 0.8]
            if len(data) == 0:
                continue
            X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar'], axis=1)[:1000]
            y_test = data['gamma_air'][:1000]
            weight_test = data['gamma_air-err'][:1000]
            errorbar = np.mean(data['errorbar'])
    
            #y_pred_o = pipe_o.predict(X_test)
            #score_o = pipe_o.score(X_test, y_test, weight_test)
    
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

            scores_full_key.append(score_full)
            scores_min1_key.append(score_min1)
            scores_min2_key.append(score_min2)
            scores_min3_key.append(score_min3)
            scores_min4_key.append(score_min4)
            scores_min5_key.append(score_min5)



            #vee = ((y_test - y_test.mean()) ** 2).sum()
            #score_n2 = score_n * vee
            print(molecule)
            print('Scores:')
            print('score = ')
            #print(score_n)
            #print('this score should be worse')
            #print(score_n2)
    
            #mse_score_o = mean_squared_error(y_test, y_pred_o)
            mse_score_full = 100 * np.sqrt(mean_squared_error(y_test, y_pred_full)) / np.mean(y_test)
            mse_score_min1 = 100 * np.sqrt(mean_squared_error(y_test, y_pred_min1)) / np.mean(y_test)
            mse_score_min2 = 100 * np.sqrt(mean_squared_error(y_test, y_pred_min2)) / np.mean(y_test)
            mse_score_min3 = 100 * np.sqrt(mean_squared_error(y_test, y_pred_min3)) / np.mean(y_test)
            mse_score_min4 = 100 * np.sqrt(mean_squared_error(y_test, y_pred_min4)) / np.mean(y_test)
            mse_score_min5 = 100 * np.sqrt(mean_squared_error(y_test, y_pred_min5)) / np.mean(y_test)



            mses_full_key.append(mse_score_full)
            mses_min1_key.append(mse_score_min1)
            mses_min2_key.append(mse_score_min2)
            mses_min3_key.append(mse_score_min3)
            mses_min4_key.append(mse_score_min4)
            mses_min5_key.append(mse_score_min5)


            print('mse = %')
            # print(mse_score_n)
    
            print(np.mean(weight_test))
            print()
            #scores_o.append(score_o)
            #scores_n_key.append(score_n2)
            mols_llist_key.append(molecule)
            #mses_key.append(mse_score_n)
    
            mean_errorbars = 100 * np.mean(data['errorbar'])
            minsss.append(mean_errorbars)
            
            err_codes = data['errorbar'][:1000].value_counts().sort_index()
            data_by_vib_lev = {}
            #print('codes!')
            #print(err_codes)
            
            figure(figsize=((15, 7)), dpi=500)
            fig = plt.figure(1)
            plt.rc('font', size=14)
            
           
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
                    label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), -int(floor(log10(abs(code*np.mean(data_level_y)))))))+f' cm$^{-1}$atm$^{-1}$'
                    plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
                    toterr += code*np.mean(data_level_y)*len(data_level_y)/1000
                #else:
                #    label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
                #    plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
        
        
                
                plt.plot(data_level_x, data_level_y, 'x', label=label, color=col)
                
                
                i+=1
        
        
            plt.plot(X_test['M'], y_pred_full, '.', color=cmap[3], label="$\gamma$ from all bits")
            plt.plot(X_test['M'], y_pred_min3, '.', color=cmap[1], label="$\gamma$ minus SVR")
            plt.plot(X_test['M'], y_pred_min5, '.', color=cmap[9], label="$\gamma$ minus mlp")
        
            
            if len(err_codes.index) == 1:
                if err_codes.index != [0.5]:
                    plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.mean(y_test)*mse_score_full/100, color=cmap[3], label='RMSE of the Voter Model')
                else:
                    toterr = 0
                    
            else:
                plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.mean(y_test)*mse_score_full/100, color=cmap[3], label='RMSE of the Voter Model')
        
        
            Air = 'Air'
            plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
            plt.xlabel('M, rotational quantum number')
            plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
            plt.ylim(0, max([max(y_pred_full), max(y_test)])+0.01)
        
            plt.legend()
            '''
            if mean_errorbars > mse_score_full:
                good += 1
                print('GOOD!')
            #elif toterr == 0:
            #    no_hit += 1
            #    print('NOHIT')
            else:
                bad += 1
                print('BAD')
            '''
            if len(err_codes.index) == 1:
                if err_codes.index == [0.5]:
                    plt.savefig('model_plot_for_paper/oldnew_comparison_plot_'+str(molecule))
                else:
                    plt.savefig('model_plot_for_paper/oldnew_comparison_plot_'+str(molecule))
            else:
                plt.savefig('model_plot_for_paper/oldnew_comparison_plot_'+str(molecule))
            
            #plt.show()
            plt.clf()
            
    
        mols_llist.append(mols_llist_key)
        scores_n.append(scores_n_key)
        kikis.append(key)
        mses_full.append(mses_full_key)
        mses_min1.append(mses_min1_key)
        mses_min2.append(mses_min2_key)
        mses_min3.append(mses_min3_key)
        mses_min4.append(mses_min4_key)
        mses_min5.append(mses_min5_key)
        
        scoreeo_full.append(scores_full_key)
        scoreeo_min1.append(scores_min1_key)
        scoreeo_min2.append(scores_min2_key)
        scoreeo_min3.append(scores_min3_key)
        scoreeo_min4.append(scores_min4_key)
        scoreeo_min5.append(scores_min5_key)
        
        
#%%
print(np.mean(mses_full))
print(np.mean(mses_min1))
print(np.mean(mses_min2))
print(np.mean(mses_min3))
print(np.mean(mses_min4))
print(np.mean(mses_min5))


#%%
print(np.mean(scoreeo_full))
print(np.mean(scoreeo_min1))
print(np.mean(scoreeo_min2))
print(np.mean(scoreeo_min3))
print(np.mean(scoreeo_min4))
print(np.mean(scoreeo_min5))

#%%

print(scoreeo_full)
print(scoreeo_min1)
print(scoreeo_min2)
print(scoreeo_min3)
print(scoreeo_min4)
print(scoreeo_min5)
#%%

mseeeeees = []
for msees in mses:
    mse = np.mean(msees)
    mseeeeees.append((mse))
d = {'model': kikis, 'average mse': mseeeeees}
df = pd.DataFrame(data=d)
#%%
print(scores_n_key)
print(mols_llist_key)
print(mses_key)
print(minsss)
#%%
print(np.mean(mses_key))
print(np.mean(minsss))
#%%
       '''
                    
                    
                    err_codes = data['errorbar'][:1000].value_counts().sort_index()
                    data_by_vib_lev = {}
                    #print('codes!')
                    #print(err_codes)
                    
                    figure(figsize=((15, 7)), dpi=500)
                    fig = plt.figure(1)
                    plt.rc('font', size=14)
                    
                   
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
                            label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), -int(floor(log10(abs(code*np.mean(data_level_y)))))))+f' cm$^{-1}$atm$^{-1}$'
                            plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
                            toterr += code*np.mean(data_level_y)*len(data_level_y)/1000
                        #else:
                        #    label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
                        #    plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
                
                
                        
                        plt.plot(data_level_x, data_level_y, 'x', label=label, color=col)
                        
                        
                        i+=1
                
                
                    #plt.plot(X_test['M'], y_pred_o, '.', color=cmap[3], label="$\gamma$ from older HITRAN data")
                    plt.plot(X_test['M'], y_pred_n, '.', color=cmap[1], label="$\gamma$ from new HITRAN data")
                
                    
                    if len(err_codes.index) == 1:
                        if err_codes.index != [0.5]:
                            plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n), color=cmap[3], label='RMSE of the Voter Model')
                        else:
                            toterr = 0
                            
                    else:
                        plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n), color=cmap[3], label='RMSE of the Voter Model')
                
                
                    Air = 'Air'
                    plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
                    plt.xlabel('M, rotational quantum number')
                    plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
                    plt.ylim(0, max([max(y_pred_n), max(y_test)])+0.01)
                
                    plt.legend()
                    
                    if toterr > np.sqrt(mse_score_n):
                        good += 1
                        print('GOOD!')
                    elif toterr == 0:
                        no_hit += 1
                        print('NOHIT')
                    else:
                        bad += 1
                        print('BAD')
                        
                    if len(err_codes.index) == 1:
                        if err_codes.index == [0.5]:
                            plt.savefig('model_plot_for_paper/oldnew_comparison_plot_'+str(molecule))
                        else:
                            plt.savefig('model_plot_for_paper/oldnew_comparison_plot_'+str(molecule))
                    else:
                        plt.savefig('model_plot_for_paper/oldnew_comparison_plot_'+str(molecule))
                    plt.show()
                    plt.clf()
                    break
                    '''

#%%
print(good)
print(bad)
#%%
print(30/48)
#%%
