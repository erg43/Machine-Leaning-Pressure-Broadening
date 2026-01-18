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
#%%
# absolute path to folder containing data
rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/full_molecules_molecule/*'
#rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/molecules_oracle/*'

# be selective for data files
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-3:] == "csv" if "readme" not in f]

# read data files, taking the filename from absolute path
db = {}
for f in file_list:
    #i = f[94:-4]
    i = f[101:-4]
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
with open('voter_model_Voter_model_molecules_all_have_diameter-no_extra_params.pkl', 'rb') as f:
    voter_models_more_no = pickle.load(f)
with open('voter_model_Voter_all_have_diameter.pkl', 'rb') as f:
    voter_models_mn_quick = pickle.load(f)
with open('voter_model_Voter_amdhdf_reweight.pkl', 'rb') as f:
    voter_models_mnqncfd_reweight = pickle.load(f)
with open('voter_model_Voter_all_have_diameter_more_data_filter.pkl', 'rb') as f:
    voter_models_mnqnc_filt_dat = pickle.load(f)
with open('voter_model_Voter_amdhdf_reweight_m.pkl', 'rb') as f:
    voter_models_mnqncfd_reweight_m = pickle.load(f)
with open('search_ML_models_mlp_alp001_3tol.pkl', 'rb') as f:
    voter_models3 = pickle.load(f)
'''
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
'''
#%%

#with open("baseline_results.json", "r") as fp:
#    list_data = json.load(fp)

with open('search_ML_models_histgrad_data.pkl', 'rb') as f:
    list_data = pickle.load(f)
with open('voter_model_best_voter_data.pkl', 'rb') as f:
    list_data_compare = pickle.load(f)
with open('voter_model_Voter_model_molecules_all_have_diameter-no_extra_params_data.pkl', 'rb') as f:
    list_data_extra_no = pickle.load(f)
with open('voter_model_Voter_all_have_diameter_data.pkl', 'rb') as f:
    list_data_mn_quick = pickle.load(f)
with open('voter_model_Voter_amdhdf_reweight_data.pkl', 'rb') as f:
    list_data_mnqncfd_reweight = pickle.load(f)
with open('voter_model_Voter_all_have_diameter_more_data_filter_data.pkl', 'rb') as f:
    list_data_mnqnc_filt_dat = pickle.load(f)
with open('voter_model_Voter_amdhdf_reweight_data_m.pkl', 'rb') as f:
    list_data_mnqncfd_reweight_m = pickle.load(f)
with open('search_ML_models_histgrad_data.pkl', 'rb') as f:
    voter_models3_d = pickle.load(f)
'''
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
'''
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
named_more_data_models_no = []
for model in voter_models_more_no:
    labels = list_data_extra_no[i][0].split(',')
    named_more_data_models_no.append([labels, model])
    i+=1
     
     
        
i = 0
mn_quick = []
for model in voter_models_mn_quick:
    labels = list_data_mn_quick[i][0].split(',') 
    mn_quick.append([labels, model])
    i+=1
        
i = 0
mnqnc_filt_dat = []
for model in voter_models_mnqnc_filt_dat:
    labels = list_data_mnqnc_filt_dat[i][0].split(',')
    mnqnc_filt_dat.append([labels, model])
    i+=1

i = 0
mnqncfd_reweight = []
for model in voter_models_mnqncfd_reweight:
    labels = list_data_mnqncfd_reweight[i][0].split(',')
    mnqncfd_reweight.append([labels, model])
    i+=1

i = 0
mnqncfd_reweight_m = []
for model in voter_models_mnqncfd_reweight_m:
    labels = list_data_mnqncfd_reweight_m[i][0].split(',')
    mnqncfd_reweight_m.append([labels, model])
    i+=1

i = 0
named_voter_models3 = []
for model in voter_models3:
    labels = voter_models3_d[i][0].split(',')
    named_voter_models3.append([labels, model])
    i+=1


#break
'''i = 0
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
    i+=1'''
#%%
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#%%
cmap = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
cmap = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
#%%
import copy
from sklearn.metrics import mean_squared_error
from math import log10, floor

good = 0
bad = 0
no_hit = 0

scores_h = []
scores_v = []
scores_mkjndsv = []
scores_a = []
scores_b = []
scores_c = []
scores_d = []
scores_e = []

mols_llist = []
colours = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
import matplotlib as mpl

for molecule in molecules:
    if molecules[molecule]['Kapp_aprox'][0] == 0:
        continue
    #if molecule != 'NO2' and molecule != 'ClO':
    #    continue
    #if molecule != 'C2H6':
    #    continue
    data = molecules[molecule]
    for item in named_hist_models:
        if molecule in item[0]:
            pipe_h = item[1]
    for item in named_voter_models:
        if molecule in item[0]:
            pipe_v = item[1]
            
    for item in mn_quick:
        if molecule in item[0]:
            pipe_mnq = item[1]

    for item in named_more_data_models_no:
        if molecule in item[0]:
            pipe_vm_no = item[1]
    for item in mnqnc_filt_dat:
        if molecule in item[0]:
            pipe_mnqncfd = item[1]
    for item in mnqncfd_reweight:
        if molecule in item[0]:
            pipe_mnqncfdrw = item[1]
    for item in mnqncfd_reweight_m:
        if molecule in item[0]:
            pipe_mnqncfdrw_m = item[1]
    for item in named_voter_models3:
        if molecule in item[0]:
            pipe_v3 = item[1]
    '''for item in named_voter_models2:
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
            #pipeer5 = item[3]'''
    
    


 
    data = data.sample(frac=1)
    X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar'], axis=1)[:1000]
    y_test = data['gamma_air'][:1000]
    weight_test = data['gamma_air-err'][:1000]
    errorbar = np.mean(data['errorbar'])

    y_pred_h = pipe_h.predict(X_test)
    score_h = pipe_h.score(X_test, y_test, weight_test)

    y_pred_v = pipe_v.predict(X_test)
    score_v = pipe_v.score(X_test, y_test, weight_test)
    
    y_pred_mnq = pipe_mnq.predict(X_test)
    score_mnq = pipe_mnq.score(X_test, y_test, weight_test)
    y_pred_mnqncfd = pipe_mnqncfd.predict(X_test)
    score_mnqncfd = pipe_mnqncfd.score(X_test, y_test, weight_test)

    y_pred_mnqncfdrw = pipe_mnqncfdrw.predict(X_test)
    score_mnqncfdrw = pipe_mnqncfdrw.score(X_test, y_test, weight_test)
    y_pred_mnqncfdrw_m = pipe_mnqncfdrw_m.predict(X_test)
    score_mnqncfdrw_m = pipe_mnqncfdrw_m.score(X_test, y_test, weight_test)
    #scores_mkjndsv.append(score_mnqncfdrw_m)

    y_pred_vm_no = pipe_vm_no.predict(X_test)
    score_vm_no = pipe_vm_no.score(X_test, y_test, weight_test)

    y_pred_v3 = pipe_v3.predict(X_test)
    score_v3 = pipe_v3.score(X_test, y_test, weight_test)



    print('Scores:')
    print('score histgrad = ')
    print(score_h)
    print('score voter = ')
    print(score_v)
    print('score new molecule diameters = ')
    print(score_mnq)
    print('score new molecule diameters more data filter data = ')
    print(score_mnqncfd)
    print('score new molecule diameters more data filter data reweight = ')
    print(score_mnqncfdrw)
    print('score new molecule diameters more data filter data reweight data m = ')
    print(score_mnqncfdrw_m)
    print('score voter model_molecules_all_have_diameter-no_extra_params = ')
    print(score_vm_no)
    print('score mlp = ')
    print(score_v3)
    print(score_h, score_v, score_mnq, score_vm_no)
    
    
    
    mse_score = mean_squared_error(y_test, y_pred_mnqncfdrw_m)
    

    scores_h.append(score_h)
    scores_v.append(score_v)
    mols_llist.append(molecule)
    
    scores_a.append(score_mnq)
    scores_b.append(score_mnqncfd)
    scores_c.append(score_mnqncfdrw)
    scores_mkjndsv.append(score_mnqncfdrw_m)
    scores_d.append(score_vm_no)
    scores_e.append(score_v3)
    

    
    print(molecule)

    

    
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    plt.rc('font', size=14)
    

    toterr = 0
    i=0

    print(data.columns)
    ks = data['Kapp_aprox'][:1000].value_counts().sort_index()
    for k, gsfifbe in ks.iteritems():
        data_level_x = X_test['Kapp_aprox'][data['Kapp_aprox'][:1000]==k]
        data_level_y_test = y_test[data['Kapp_aprox'][:1000]==k]
        data_level_y_pred = y_pred_mnqncfdrw_m[data['Kapp_aprox'][:1000]==k]
        plt.plot(data_level_x, data_level_y_pred/data_level_y_test, '.', label="$\gamma$ predicted by Voter Model, k = "+str(k))
        
    plt.axhline(y=1, xmin=min(X_test['M']), xmax=max(X_test['M']))


    Air = 'Air'
    plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {molecule}')
    plt.xlabel('K')
    plt.ylabel(f'pred/test Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')

    #plt.legend()
    
    if toterr > np.sqrt(mse_score):
        good += 1
        print('GOOD!')
    elif toterr == 0:
        no_hit += 1
        print('NOHIT')
    else:
        bad += 1
        print('BAD')
    
    

    plt.show()
    
    #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    
    err_codes = data['errorbar'][:1000].value_counts().sort_index()
    data_by_vib_lev = {}

    
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

        
        #if len(err_codes.index) == 1:
        #print(err_codes.index)
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
        
        #print('amount data =')
        #print(len(data_level_y)/1000)
        
        i+=1


    #plt.plot(X_test['M'], y_pred_v2, '.', color=cmap[2], label="voter model with mlp excluded, Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v4, '.', color=cmap[0], label="adaboost model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v5, '.', color=cmap[8], label="random forest model Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_v3, '.', color=cmap[9], label="$\gamma$ predicted by Multilayer Perceptron Model")
    #plt.plot(X_test['M'], y_pred_h, '.', color=cmap[1], label="$\gamma$ predicted by Gradient Boosting Model")
    #plt.plot(X_test['M'], y_pred_v, '.', color=cmap[3], label="$\gamma$ predicted by Voter Model")
    #plt.plot(X_test['M'], y_pred_vm_no, '.', color=cmap[0], label="voter no fluff Predicted $\gamma$")
    #plt.plot(X_test['M'], y_pred_mnq, '.', color=cmap[9], label="voter, no fluff, quicker Predicted $\gamma$")
    plt.plot(X_test['M'], y_pred_mnqncfdrw_m, '.', color=cmap[3], label="$\gamma$ predicted by Voter Model")
    #plt.plot(X_test['M'], y_pred_mnqncfdrw, '.', color=cmap[8], label="everything reweight Predicted $\gamma$")


    
    if len(err_codes.index) == 1:
        if err_codes.index != [0.5]:
            plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score), color=cmap[3], label='RMSE of the Voter Model')
        else:
            toterr = 0
            
    else:
        plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score), color=cmap[3], label='RMSE of the Voter Model')


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
    plt.ylim(0, max([max(y_pred_mnqncfdrw_m), max(y_test)])+0.01)
    #plt.ylim(0, 0.15)
    #print(errorbar)

    #plt.rcParams.update({'font.size': 14})
    
    #plt.ylim(0)
    plt.legend()
    plt.show()
    

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
socres = {'histgrad': scores_h, 'voting': scores_mkjndsv}
df = pd.DataFrame(socres)
df = df.set_index([pd.Index(mols_llist)])
df['percentage difference'] = 100* (df['voting'] - df['histgrad'])/(df['histgrad']-1).abs()
df
#%%
print(df['percentage difference'].sum())
#%%
CS2
CS
COF2
NO2
O3
HBr
O2
PH3
HNO3
N2
HO2
SO2
COCl2
HC3N
CH3OH
OCS
HOBr
CH3I
H2O2
H2S
CH3Cl
HF
NH3
H2
CH3CN
SO
HI
ClO
CH3Br
C2N2
N2O
HOCl
H2CO
CO2
H2O
CH3F
HCOOH
C2H2
CO
C2H4
OH
HCl
C4H2
HCN
C2H6
CH4
NO
#%%
print('Scores:')
print('score histgrad = ')
print(scores_h)
print('score voter = ')
print(scores_v)
print('score new molecule diameters = ')
print(scores_a)
print('score new molecule diameters more data filter data = ')
print(scores_b)
print('score new molecule diameters more data filter data reweight = ')
print(scores_c)
print('score new molecule diameters more data filter data reweight data m = ')
print(scores_mkjndsv)
print('score voter model_molecules_all_have_diameter-no_extra_params = ')
print(scores_d)
print('score mlp = ')
print(scores_e)