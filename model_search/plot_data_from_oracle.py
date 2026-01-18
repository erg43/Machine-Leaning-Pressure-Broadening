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
import numpy as np
import pickle
#with open("baseline_results.json", "r") as fp:
#    list_data = json.load(fp)

with open('search_ML_models_mlp_data.pkl', 'rb') as f:
    list_data = pickle.load(f)

for item in list_data:
    for mass in np.unique(item[6]):
            data_by_mass_j = item[1][:1000][item[6][:1000]==mass]
            data_by_mass_hit = item[2][:1000][item[6][:1000]==mass]
            data_by_mass_ml = item[3][:1000][item[6][:1000]==mass]
            label = masses[str(mass)]
            print(label)
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

with open('search_ML_models_adaboost_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()

for item in list_data:
    print(item[4])



for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_histgrad_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()

for item in list_data:
    print(item[4])



for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_sgd_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()

for item in list_data:
    print(item[4])



for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_linear_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()


for item in list_data:
    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_dummy_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()


for item in list_data:
    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_randomforest_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()



for item in list_data:
    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_decision_tree1_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()


for item in list_data:
    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_decision_tree2_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()


for item in list_data:
    print(item[4])



for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_decision_tree3_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()


for item in list_data:
    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_decision_tree4_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()


for item in list_data:
    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_decision_tree9_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()


for item in list_data:
    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_svr_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()

for item in list_data:
    print(item[4])



for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_mlp_data.pkl', 'rb') as f:
    list_data = pickle.load(f)

#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()

for item in list_data:

    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    for mass in np.unique(item[6]):
        data_by_mass_j = item[1][:1000][item[6][:1000]==mass]
        data_by_mass_hit = item[2][:1000][item[6][:1000]==mass]
        data_by_mass_ml = item[3][:1000][item[6][:1000]==mass]
        label = masses[str(mass)]

        plt.plot(data_by_mass_j, data_by_mass_hit, 'x', label='HITRAN $\gamma$ data')
        plt.plot(data_by_mass_j, data_by_mass_ml, 'o', label="Predicted $\gamma$")
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
    plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {key}')
    plt.xlabel('M, rotational quantum number')
    plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
    #plt.ylim(0)
    plt.legend()
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_mlp_better_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()


for item in list_data:
    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('search_ML_models_mlp_best_data.pkl', 'rb') as f:
    list_data = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()


for item in list_data:
    print(item[4])


for item in list_data:
    key = item[0]
    print(item[4])
    # make plots
    figure(figsize=((15, 7)), dpi=500)
    fig = plt.figure(1)
    #for code in err_codes.index:
    #    data_level_x = data_test[data_test['gamma_air-err']==code]
    #    label = str(np.sqrt(code))
    plt.plot(item[1], item[2], 'x', label='HITRAN $\gamma$ data')
    plt.plot(item[1], item[3], 'o', label="Predicted $\gamma$")
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
    
    #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
    plt.show()
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

with open('voter_model_histgrad_weighting_embedded_in_data.pkl', 'rb') as f:
    list_data = pickle.load(f)
#with open('search_ML_models_mlp_alp001_3tol_data.pkl', 'rb') as f:
with open('voter_model_voter_full_weighting_data.pkl', 'rb') as f:
    list_data_compare = pickle.load(f)


#list_data = pd.read_csv('baseline.csv')


#file = open("baseline.csv", "r")
#list_data = list(csv.reader(file, delimiter=","))
#file.close()

for item in list_data:

    print(item[4])

count=0
for item in list_data:
    key = item[0]
    comparison_data = list_data_compare[count]
    count+=1

    print('hist has score of'+str(item[4]))
    print('new has score of '+str(comparison_data[4]))
    # make plots
    
    for mass in np.unique(item[6]):
        data_by_mass_j = item[1][item[6]==mass]
        data_by_mass_hit = item[2][item[6]==mass]
        data_by_mass_ml = item[3][item[6]==mass]
        label = masses[str(mass)]
        
        
        comp_data_by_mass_j = comparison_data[1][comparison_data[6]==mass]
        comp_data_by_mass_hit = comparison_data[2][comparison_data[6]==mass]
        comp_data_by_mass_ml = comparison_data[3][comparison_data[6]==mass]
        
        
        figure(figsize=((15, 7)), dpi=500)
        fig = plt.figure(1)
        plt.plot(data_by_mass_j, data_by_mass_hit, 'x', label='hist HITRAN $\gamma$ data for '+label)
        plt.plot(data_by_mass_j, data_by_mass_ml, 'o', label="hist Predicted $\gamma$ for "+label)
        
        plt.plot(comp_data_by_mass_j, comp_data_by_mass_hit, 'x', label='new HITRAN $\gamma$ data for '+label)
        plt.plot(comp_data_by_mass_j, comp_data_by_mass_ml, 'o', label="new Predicted $\gamma$ for "+label)
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
        plt.title(f'Comparison of $\gamma_{{{Air}}}$ from machine learning results against HITRAN data values, shown for {label}')
        plt.xlabel('M, rotational quantum number')
        plt.ylabel(f'Line broadening, $\gamma_{Air}$ /cm$^{-1}$atm$^{-1}$')
        #plt.ylim(0)
        plt.legend()
        
        #plt.savefig(f'ML_results_no_mol_weight/{key}.png')
        plt.show()
#%%
