import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pickle
import copy



### Read in data

# import data
path = ""
file_list = glob.glob(path + 'data/*.csv')


# read data files, taking the filename from absolute path
molecules = {}
for f in file_list:
    i = f[4:-4]
    molecules[i] = pd.read_csv(f, low_memory=False)



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
    

with open('trained_models.pkl', 'rb') as f:
    models = pickle.load(f)

i = 0
models_list = []
for model in models:
    labels = model[0].split(',')
    models_list.append([labels, model[1]])

cmap = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

for molecule in molecules:
    data = molecules[molecule]
    for item in models_list:
        if molecule in item[0]:
            pipe = item[1]
    
    data.sample(frac=1)
    data2 = data[data['errorbar'] != 0.5]
    data2 = data2[data2['errorbar'] != 0.6]
    data2 = data2[data2['errorbar'] != 0.7]
    data2 = data2[data2['errorbar'] != 0.8]
    if len(data2) != 0:
        X_test2 = data2.drop(['gamma_air', 'gamma_air-err', 'errorbar', 'code'], axis=1)[:500]
        y_test2 = data2['gamma_air'][:500]
        y_pred_n2 = pipe_n.predict(X_test2)
        mse_score_n2 = mean_squared_error(y_test2, y_pred_n2)
        
    
    X_test = data.drop(['gamma_air', 'gamma_air-err', 'errorbar', 'code'], axis=1)[:500]
    y_test = data['gamma_air'][:500]
    weight_test = data['gamma_air-err'][:500]
    errorbar = np.mean(data['errorbar'])
    codeyy = data['code'][:500]

    y_pred = pipe.predict(X_test)
    score = pipe.score(X_test, y_test, weight_test)
    mse_score = mean_squared_error(y_test, y_pred)
    
    err_codes = data['errorbar'][:500].value_counts().sort_index()
    data_by_vib_lev = {}
    
    figure(figsize=((15,6)), dpi=500)
    fig = plt.figure(1)
    plt.rc('font', size=18)
    
    i=0
    for code, gsfifbe in err_codes.iteritems():
        if molecule == 'C2H6':
            if code == 0.2:
                continue
        data_level_x = X_test['M'][data['errorbar']==code]
        data_level_y = y_test[data['errorbar']==code]
        
                
        if code==0.01:
            coode = 8
            col=cmap[7]
        elif code==0.02:
            coode = 7
            col=cmap[11]
        elif code==0.05:
            coode = 6
            col=cmap[9]
        elif code==0.1:
            coode = 5
            col=cmap[2]
        elif code==0.2:
            coode = 4
            col=cmap[0]
        elif code==0.5:
            coode = 3
            col=cmap[4]
        elif code==0.6:
            coode = 2
            col=cmap[4]
        elif code==0.7:
            coode = 1
            col=cmap[4]
        elif code==0.8:
            coode = 0
            col=cmap[4]

        
        if code in [0.5, 0.6, 0.7, 0.8]:
            label = 'Error code = '+str(coode)
        else:
            label = 'Error code = '+str(coode)
            if molecule == 'HCl':
                plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test)+0.02, yerr=code*np.mean(data_level_y), color=col)
            else:
                plt.errorbar(max(X_test['M']+3+3*i), np.mean(y_test), yerr=code*np.mean(data_level_y), color=col)
        
        plt.plot(data_level_x, data_level_y, 'x', label=label, color=col)
        
        i+=1

    plt.plot(X_test['M'], y_pred, '.', color=cmap[5], label="$\gamma$ predicted by Voting Model")
    plt.hlines(X_test['broadness_jeanna'].iloc[0], min(X_test['M']), max(X_test['M']), linewidth=2, color=cmap[1], label="$\gamma$ predicted by Buldyreva")
    
    if len(data2) != 0:
        if len(err_codes.index) == 1:
            if err_codes.index != [0.5]:
                plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n2), color=cmap[5], label='RMSE of the Voting Model')
            else:
                toterr = 0
        else:
            if molecule == 'HCl':
                plt.errorbar(min(X_test['M']-3), np.mean(y_test)+0.02, yerr=np.sqrt(mse_score_n2), color=cmap[5], label='RMSE of the ML Model')
            else:
                plt.errorbar(min(X_test['M']-3), np.mean(y_test), yerr=np.sqrt(mse_score_n2), color=cmap[5], label='RMSE of the Voting Model') 
        
    if len(err_codes.index) == 1 and err_codes.index == [0.5]:
        toterr = 0
    
    Air = 'Air'
    plt.xlabel('m')
    plt.ylabel(f'$\gamma_{{{Air}}}$ /cm$^{{{-1}}}$atm$^{{{-1}}}$')
    plt.ylim(0.00, max([max(y_pred_n), max(y_test), max(y_pred_m), max(y_pred_f)])+0.01)
    
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2)
    plt.show()
