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
rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/raw_data/*'
#rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/molecules_oracle/*'

# be selective for data files
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-3:] == "csv" if "readme" not in f]

#file_list.insert(0, file_list.pop(49))
# read data files, taking the filename from absolute path
db = {}
for f in file_list:
    #i = f[94:-4]
    i = f[86:-4]
    print(i)
    #if i == 'C2H6':
    db[i] = pd.read_csv(f)
#%%
# Set parameters
T = 298      # Kelvin


# define Jeanna's broadening formalism for use later.  Taken from paper ...
def broadening(m, T, ma, mp, b0):
    gamma = 1.7796e-5 * (m/(m-2)) * (1/np.sqrt(T)) * np.sqrt((ma+mp)/(ma*mp)) * b0**2
    return(gamma)



#%%
dif_sum = []
    
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
        
        
        #if key == 'C2H6':
        #    data = data.drop(data[(data['J'] > 20) & (data['gamma_air'] > 0.08)].index)
        
        
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
        
        print(key)
        # calculate jeanna broadness, and add to dictionary
        if key == 'CH3CN':
            broadness_jeanna = broadening(data['m'][20], T, data['molecule_weight'][20], data['air_weight'][20], data['findair'][20])
        elif key == 'SO3':
            broadness_jeanna = broadening(data['m'][258], T, data['molecule_weight'][258], data['air_weight'][258], data['findair'][258])
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
    molecules[molecule]['code'] = weight
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
with open('voter_model_final_10-25_hist.pkl', 'rb') as f:
    hist_model = pickle.load(f)
with open('voter_model_final_10-25_mlp.pkl', 'rb') as f:
    mlp_model = pickle.load(f)
with open('voter_model_final_10-27_voter.pkl', 'rb') as f:
    new_model = pickle.load(f)
with open('voter_model_final_10-25_forest.pkl', 'rb') as f:
    for_model = pickle.load(f)
#%%
#with open('search_ML_models_histgrad_data.pkl', 'rb') as f:
#    hist_data = pickle.load(f)
#with open('voter_model_best_histgrad_data.pkl', 'rb') as f:
#    hist_data = pickle.load(f)
#with open('search_ML_models_mlp_alp001_3tol_data.pkl', 'rb') as f:
#    mlp_data = pickle.load(f)
#with open('search_ML_models_mlp_best_data.pkl', 'rb') as f:
#    mlp_data = pickle.load(f)
with open('voter_model_final_10-25_hist_data.pkl', 'rb') as f:
    hist_data = pickle.load(f)
with open('voter_model_final_10-25_mlp_data.pkl', 'rb') as f:
    mlp_data = pickle.load(f)
with open('voter_model_final_10-25_forest_data.pkl', 'rb') as f:
    for_data = pickle.load(f)

with open('voter_model_final_10-27_voter_data.pkl', 'rb') as f:
    new_data = pickle.load(f)


#%%
i = 0
hist_models_list = []
for model in hist_model:
    labels = hist_data[i][0].split(',')
    print(labels)
    hist_models_list.append([labels, model])
    i+=1
   
print()

i = 0
for_models_list = []
for model in for_model:
    labels = for_data[i][0].split(',')
    print(labels)
    for_models_list.append([labels, model])
    i+=1
   
print()

   
i = 0
mlp_models_list = []
for model in mlp_model:
    labels = mlp_data[i][0].split(',')
    print(labels)
    print(model)
    mlp_models_list.append([labels, model])
    i+=1

print()
     
i = 0
new_models_list = []
for model in new_model:
    labels = new_data[i][0].split(',')
    print(labels)
    print(model)
    new_models_list.append([labels, model])
    i+=1

#%%
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#%%
cmap = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
cmap = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
#%%
import matplotlib
#%%
import copy
from sklearn.metrics import mean_squared_error
from math import log10, floor

good = 0
bad = 0
no_hit = 0

scores_o = []
scores_n = []

moooools = []

mse_scores_b = []
mse_scores_as = []
mse_scores_ps = []
mse_scores_os = []
mse_scores_sp = []
mse_scores_li = []
mse_scores_tot = []
mse_scores_asm = []
mse_scores_psm = []
mse_scores_osm = []
mse_scores_spm = []
mse_scores_lim = []
mse_scores_tot2 = []

mols_llist = []
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
import matplotlib as mpl


data_for_jeanna = {}

for molecule in molecules:
    #if molecule != 'NO2' and molecule != 'ClO':
    #    continue
    #if molecule != 'NO':
    #    continue
    #if molecule != ('HCl'):
    #    continue
    data = molecules[molecule]
    for item in hist_models_list:
        if molecule in item[0]:
            pipe_h = item[1]
            print(item)
    for item in for_models_list:
        if molecule in item[0]:
            pipe_f = item[1]
            print(item)
    for item in mlp_models_list:
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
    #data = data.loc[data['M'].drop_duplicates().index]
    print(len(data))
    data2 = data[data['errorbar'] != 0.5]
    data2 = data2[data2['errorbar'] != 0.6]
    data2 = data2[data2['errorbar'] != 0.7]
    data2 = data2[data2['errorbar'] != 0.8]
    if len(data2) != 0:

        X_test2 = data2.drop(['gamma_air', 'gamma_air-err', 'errorbar', 'code'], axis=1)[:500]
        y_test2 = data2['gamma_air'][:500]
        y_pred_n2 = pipe_n.predict(X_test2)
        mse_score_n2 = mean_squared_error(y_test2, y_pred_n2)
        
    

    
    #X_test = data[:500]
    #print(X_test.columns)
    
    X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar', 'code'], axis=1)[:500]
    y_test = data['gamma_air'][:500]
    weight_test = data['gamma_air-err'][:500]
    errorbar = np.mean(data['errorbar'])
    codeyy = data['code'][:500]

    y_pred_h = pipe_h.predict(X_test)
    score_h = pipe_h.score(X_test, y_test, weight_test)
    y_pred_m = pipe_m.predict(X_test)
    score_m = pipe_m.score(X_test, y_test, weight_test)
    
    y_pred_f = pipe_f.predict(X_test)
    score_f = pipe_f.score(X_test, y_test, weight_test)
    
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
 
    mse_scores_b.append(np.sqrt(mean_squared_error(y_test, X_test['broadness_jeanna']))*100/np.mean(y_test))

    
    
    err_codes = data['errorbar'][:500].value_counts().sort_index()
    data_by_vib_lev = {}
    #print('codes!')
    #print(err_codes)
    
    figure(figsize=((15,6)), dpi=500)
    fig = plt.figure(1)
    plt.rc('font', size=18)
    
    label = ''
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
            coode = '<1%'
            col=cmap[7]
        elif code==0.02:
            coode = '1 - 2%'
            col=cmap[11]
        elif code==0.05:
            coode = '2 - 5%'
            col=cmap[9]
        elif code==0.1:
            coode = '5 - 10%'
            col=cmap[2]
        elif code==0.2:
            coode = '10 - 20%'
            col=cmap[0]
        elif code==0.5:
            coode = '≥20%'
            col=cmap[4]
        elif code==0.6:
            coode = 'Average or estimate'
            col=cmap[4]   #6
        elif code==0.7:
            coode = 'Default or constant'
            col=cmap[4]   #10
        elif code==0.8:
            coode = 'Unreported or unavailable'
            col=cmap[4]   #1
        
        #print(code)
        if code in [0.5, 0.6, 0.7, 0.8]:
            if label == 'HITRAN data, uncertainty unbounded':
                label = ''
            else:
                label = 'HITRAN data, uncertainty unbounded'
            #label = 'HITRAN data, broadening values estimated from no empirical data'#, code = '+str(code)
            toterr += code*np.mean(data_level_y)*len(data_level_y)/500
        else:
            label = 'HITRAN data, uncertainty '+str(coode)
            #label = 'HITRAN data, error given by errorbar'
            #label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), -int(floor(log10(abs(code*np.mean(data_level_y)))))))+f' cm$^{-1}$atm$^{-1}$'
            if molecule == 'HCl':
                plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test)+0.02, yerr=code*np.mean(data_level_y), color=col)
            else:
                plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
            toterr += code*np.mean(data_level_y)*len(data_level_y)/500
        #else:
        #    label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
        #    plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)


        
        plt.plot(data_level_x, data_level_y, 'x', label=label, color=col)
        
        
        i+=1

    
    #plt.plot(X_test['M'], y_pred_h, '.', color=cmap[8], label="$\gamma$ predicted by Histgrad Model")
    #plt.plot(X_test['M'], y_pred_f, '.', color=cmap[8], label="$\gamma$ predicted by Random Forest Model")

    #plt.plot(X_test['M'], y_pred_m, '.', color=cmap[3], label="$\gamma$ predicted by MLP Model")
    plt.plot(X_test['M'], y_pred_n, '.', color=cmap[5], label="$\gamma$ predicted by ML Model")
    #plt.plot(X_test['M'], X_test['broadness_jeanna'], '.', color=cmap[2], label="$\gamma$ predicted by Buldyreva")
    #plt.hlines(X_test['broadness_jeanna'].iloc[0], min(X_test['M']), max(X_test['M']), linewidth=2, color=cmap[1], label="$\gamma$ predicted by Buldyreva")
    #yo = 0
    #for colour in cmap:
    #    plt.hlines(X_test['broadness_jeanna'].iloc[0]-0.005*yo, min(X_test['M']), max(X_test['M']), linewidth=4, color=colour, label="$\gamma$ predicted by Buldyreva")
    #    yo+=1

    
    if len(data2) != 0:
        if len(err_codes.index) == 1:
            if err_codes.index != [0.5]:
                plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n2), color=cmap[5], label='RMSE of the ML Model')
            else:
                toterr = 0
            
        else:
            if molecule == 'HCl':
                plt.errorbar(min(X_test['M']-3), np.mean(y_test)+0.02, yerr=np.sqrt(mse_score_n2), color=cmap[5], label='RMSE of the ML Model')
            else:
                plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n2), color=cmap[5], label='RMSE of the ML Model') 
        
        if molecule in ['C2H4', 'CH3OH', 'COF2', 'H2CO', 'H2O2', 'H2O', 'HO2', 'NO2', 'O3', 'SO2', 'COCl2', 'H2S', 'HOBr', 'HNO3', 'HOCl', 'COF2', 'NO2']:
            mse_scores_as.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/np.mean(y_test))
            mse_scores_asm.append(np.mean(y_test))
        elif molecule in ['C2H6', 'CH3F', 'CH3CN', 'CH3I', 'CH3Br', 'CH3Cl']:
            mse_scores_ps.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/np.mean(y_test))
            mse_scores_psm.append(np.mean(y_test))
        elif molecule in ['NH3', 'PH3']:
            mse_scores_os.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/np.mean(y_test))
            mse_scores_osm.append(np.mean(y_test))
        elif molecule in ['CH4']:
            mse_scores_sp.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/np.mean(y_test))
            mse_scores_spm.append(np.mean(y_test))
        elif molecule in ['OCS', 'OH', 'NO', 'N2O', 'HF', 'HCl', 'HBr', 'CS', 'CO', 'ClO', 'C2H2', 'CO2', 'H2', 'N2', 'O2', 'C2N2', 'C4H2', 'HCN', 'SO', 'HC3N', 'CS2', 'HI']:
            mse_scores_li.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/np.mean(y_test))
            mse_scores_lim.append(np.mean(y_test))
            #moooools.append(molecule)
        mse_scores_tot.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/np.mean(y_test))
        mse_scores_tot2.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/np.mean(y_pred_n))
            

    if len(err_codes.index) == 1 and err_codes.index == [0.5]:
        toterr = 0
    
    
    
    Air = 'Air'
    #plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
    plt.xlabel('m')
    plt.ylabel(f'$\gamma_{{{Air}}}$ /cm$^{{{-1}}}$atm$^{{{-1}}}$')
    plt.ylim(0.00, max([max(y_pred_n), max(y_test), max(y_pred_m), max(y_pred_f)])+0.01)
    
    
    matplotlib.rc_file_defaults()
    plt.rc('font', size=18)
    plt.grid()
    
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2)
        
    if toterr > np.sqrt(mse_score_n):
        good += 1
        print('GOOD!')
    elif toterr == 0:
        no_hit += 1
        print('NOHIT')
    else:
        bad += 1
        moooools.append(molecule)

        print('BAD')
    
    if len(err_codes.index) == 1:
        if err_codes.index == [0.5]:
            plt.savefig('model_plot_for_paper/Paper_plot_'+str(molecule)+'.pdf')
        else:
            plt.savefig('model_plot_for_paper/Paper_plot_'+str(molecule)+'.pdf')
    else:
        plt.savefig('model_plot_for_paper/Paper_plot_'+str(molecule)+'.pdf')
    
    #.show()
    print(molecule)
    plt.clf()
    
    
    if molecule in ['C2H4', 'CH3OH', 'COF2', 'H2CO', 'H2O2', 'H2O', 'HO2', 'NO2', 'O3', 'SO2', 'COCl2', 'H2S', 'HOBr', 'HNO3', 'HOCl', 'COF2', 'NO2']:
        mse_scores_as.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/0.07461589611111112)#np.mean(y_test))
        mse_scores_asm.append(np.mean(y_test))
    elif molecule in ['C2H6', 'CH3F', 'CH3CN', 'CH3I', 'CH3Br', 'CH3Cl']:
        mse_scores_ps.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/0.07461589611111112)#np.mean(y_test))
        mse_scores_psm.append(np.mean(y_test))
    elif molecule in ['NH3', 'PH3']:
        mse_scores_os.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/0.07461589611111112)#np.mean(y_test))
        mse_scores_osm.append(np.mean(y_test))
    elif molecule in ['CH4']:
        mse_scores_sp.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/0.07461589611111112)#np.mean(y_test))
        mse_scores_spm.append(np.mean(y_test))
    elif molecule in ['OCS', 'OH', 'NO', 'N2O', 'HF', 'HCl', 'HBr', 'CS', 'CO', 'ClO', 'C2H2', 'CO2', 'H2', 'N2', 'O2', 'C2N2', 'C4H2', 'HCN', 'SO', 'HC3N', 'CS2', 'HI']:
        mse_scores_li.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/0.07461589611111112)#np.mean(y_test))
        mse_scores_lim.append(np.mean(y_test))
        #moooools.append(molecule)
    mse_scores_tot.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/0.07461589611111112)#np.mean(y_test))
    mse_scores_tot2.append(np.sqrt(mean_squared_error(y_test, y_pred_n))*100/0.07461589611111112)#np.mean(y_pred_n))




    
    data_for_jeanna[molecule] = {'m':X_test['M'], 'ML_gamma_prediction':y_pred_n, 'HITRAN_gamma_air':y_test, 'HITRAN_gamma_air-err':data['code'][:500], 'Buldyreva_gamma':X_test['broadness_jeanna']}

#%%
for molecule, data in data_for_jeanna.items():
    dats = pd.DataFrame(data)
    dats.to_csv('Jeanna_files/'+molecule+'_data_ML_HITRAN_Buldyreva.csv')
#%%
print(moooools)
#%%
print(np.mean(mse_scores_b), np.mean(mse_scores_as), np.mean(mse_scores_ps), np.mean(mse_scores_os), np.mean(mse_scores_sp), np.mean(mse_scores_li))
#%%
print(np.mean(mse_scores_asm)/x, np.mean(mse_scores_psm)/x, np.mean(mse_scores_osm)/x, np.mean(mse_scores_spm)/x, np.mean(mse_scores_lim)/x)
#%%
x = np.mean(mse_scores_asm)*17/48 + np.mean(mse_scores_psm)*6/48 + np.mean(mse_scores_osm)*2/48 + np.mean(mse_scores_spm)*1/48 + np.mean(mse_scores_lim)*22/48
#%%
print(np.mean(mse_scores_asm)*17/48 + np.mean(mse_scores_psm)*6/48 + np.mean(mse_scores_osm)*2/48 + np.mean(mse_scores_spm)*1/48 + np.mean(mse_scores_lim)*22/48)
#%%
print(np.mean(mse_scores_tot))
#%%

print(mse_scores_b, mse_scores_as, mse_scores_ps, mse_scores_os, mse_scores_sp, mse_scores_li)
#%%
print(np.mean(mse_scores_tot), np.mean(mse_scores_tot2))
#%%
print(mse_scores_li)
print(moooools)
#%%
print(good, bad, no_hit)
#%%
# CODE FOR DOING HONO

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
    if molecule == 'HONO_prediction':
        continue
    print(molecule)
    data = molecules[molecule]
    '''for item in hist_models_list:
        if molecule in item[0]:
            pipe_h = item[1]
            print(item)
    for item in mlp_models_list:
        if molecule in item[0]:
            print(item)
            pipe_m = item[1]'''
    for item in new_models_list:
        if molecule in item[0]:
            #print(item)
            pipe_n = item[1]

    #if molecule == 'HONO_prediction':
    #    continue
    #print(molecule)    
 
    data = data.sample(frac=1)
    
    data2 = data[data['errorbar'] != 0.5]
    data2 = data2[data2['errorbar'] != 0.6]
    data2 = data2[data2['errorbar'] != 0.7]
    data2 = data2[data2['errorbar'] != 0.8]
    if len(data2) != 0:

        X_test2 = data2.drop(['gamma_air', 'gamma_air-err', 'errorbar', 'code'], axis=1)[:1000]
        y_test2 = data2['gamma_air'][:1000]
        y_pred_n2 = pipe_n.predict(X_test2)
        mse_score_n2 = mean_squared_error(y_test2, y_pred_n2)
        
    

    
    #X_test = data[:1000]
    #print(X_test.columns)
    
    X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar', 'code'], axis=1)[:1000]
    y_test = data['gamma_air'][:1000]
    weight_test = data['gamma_air-err'][:1000]
    errorbar = np.mean(data['errorbar'])
    codeyy = data['code'][:1000]

    '''y_pred_h = pipe_h.predict(X_test)
    score_h = pipe_h.score(X_test, y_test, weight_test)
    y_pred_m = pipe_m.predict(X_test)
    score_m = pipe_m.score(X_test, y_test, weight_test)'''
    
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
    
    figure(figsize=((25,10)), dpi=500)
    fig = plt.figure(1)
    plt.rc('font', size=42)
    
    
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
            coode = 8
            col=cmap[10]
        elif code==0.02:
            coode = 7
            col=cmap[6]
        elif code==0.05:
            coode = 6
            col=cmap[7]
        elif code==0.1:
            coode = 5
            col=cmap[4]
        elif code==0.2:
            coode = 4
            col=cmap[5]
        elif code==0.5:
            coode = 3
            col=cmap[11]
        elif code==0.6:
            coode = 2
            col=cmap[11]
        elif code==0.7:
            coode = 1
            col=cmap[11]
        elif code==0.8:
            coode = 0
            col=cmap[11]

        
        #print(code)
        if code in [0.5, 0.6, 0.7, 0.8]:
            label = 'Error code = '+str(coode)
            #label = 'HITRAN data, broadening values estimated from no empirical data'#, code = '+str(code)
            toterr += code*np.mean(data_level_y)*len(data_level_y)/1000
        else:
            label = 'Error code = '+str(coode)
            #label = 'HITRAN data, error given by errorbar'
            #label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), -int(floor(log10(abs(code*np.mean(data_level_y)))))))+f' cm$^{-1}$atm$^{-1}$'
            plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
            toterr += code*np.mean(data_level_y)*len(data_level_y)/1000
        #else:
        #    label = 'HITRAN data, with error of '+str(round(code*np.mean(data_level_y), 4))+f' cm$^{-1}$atm$^{-1}$'
        #    plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)


        
        plt.plot(data_level_x, data_level_y, 'x', label=label, color=col)
        
        
        i+=1

    
    #plt.plot(X_test['M'], y_pred_h, '.', color=cmap[1], label="$\gamma$ predicted by histgrad model")
    #plt.plot(X_test['M'], y_pred_m, '.', color=cmap[5], label="$\gamma$ predicted by mlp model")
    plt.plot(X_test['M'], y_pred_n, '.', color=cmap[3], label="$\gamma$ predicted by voting model")
    
    if len(data2) != 0:
        if len(err_codes.index) == 1:
            if err_codes.index != [0.5]:
                plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n2), color=cmap[3], label='RMSE of voting Model')
            else:
                toterr = 0
            
        else:
            plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n2), color=cmap[3], label='RMSE of the ML Model')

    if len(err_codes.index) == 1 and err_codes.index == [0.5]:
        toterr = 0
    
    
    Air = 'Air'
    #plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
    plt.xlabel('m')
    plt.ylabel(f'$\gamma_{{{Air}}}$ /cm$^{{{-1}}}$atm$^{{{-1}}}$')
    plt.ylim(0.00, max([max(y_pred_n), max(y_test)])+0.01)
    
    
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
        
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
    #        plt.savefig('model_plot_for_paper/Paper_nohit_plot_'+str(molecule))
    #    else:
    #        plt.savefig('model_plot_for_paper/Paper_plot_'+str(molecule))
    #else:
    #    plt.savefig('model_plot_for_paper/Paper_plot_'+str(molecule))
    
    plt.show()
    plt.savefig('trial')
    #plt.clf()
    break
    
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

    #if molecule != ('HONO_prediction'):
    #    continue
    data = molecules[molecule]
    print(molecule)
    for item in new_models_list:
        if molecule in item[0]:
            #print(item)
            pipe_n = item[1]
    for item in hist_models_list:
        if molecule in item[0]:
            #print(item)
            pipe_h = item[1]
    for item in mlp_models_list:
        if molecule in item[0]:
            #print(item)
            pipe_m = item[1]

    #if molecule == 'HONO_prediction':
    #    continue
    #print(molecule)    

    data = data.sample(frac=1)


    #X_test = data[:1000]
    #print(X_test.columns)

    X_test = data[:1000]
    y_pred_n = pipe_n.predict(X_test)
    
    mols_llist.append(molecule)

    figure(figsize=((18, 10)), dpi=500)
    fig = plt.figure(1)
    plt.rc('font', size=42)

    
    



    plt.plot(X_test['M'], y_pred_n, '.', color=cmap[3])


    Air = 'Air'
    plt.xlabel('m')
    plt.ylabel(f'$\gamma_{{{Air}}}$ /cm$^{{{-1}}}$atm$^{{{-1}}}$')
    plt.ylim(0.00, max([max(y_pred_n)]) + 0.01)

    #ax = plt.subplot(111)
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)


    #plt.savefig('model_plot_for_paper/Paper_plot_' + str(molecule))

    plt.show()

    #plt.clf()
    continue
#%%
