#%%
import pandas as pd
import numpy as np

co = pd.read_csv("CO/1_iso.csv")
#o2 = pd.read_csv("O2/O2_1_iso.csv")
no = pd.read_csv("NO/1_iso.csv")
oh = pd.read_csv("OH/1_iso.csv")
hf = pd.read_csv("HF/1_iso.csv")
hcl = pd.read_csv("HCl/1_iso.csv")
hbr = pd.read_csv("HBr/1_iso.csv")
hi = pd.read_csv("HI/1_iso.csv")
clo = pd.read_csv("ClO/1_iso.csv")
#n2 = pd.read_csv("N2/N2_1_iso.csv")
#h2 = pd.read_csv("H2/H2_1_iso.csv")
cs = pd.read_csv("CS/1_iso.csv")
so = pd.read_csv("SO/1_iso.csv")

#%%

#%%
molecules = {"CO": co, "NO": no, "OH": oh,"HF": hf, "HCl": hcl, "HBr": hbr, "HI": hi, "ClO": clo, "CS": cs, "SO": so}
#%%
import random

test_data = random.choices(list(molecules), k=1)
print(test_data)
train_data = set(molecules) - set(test_data)
print(train_data)

test_data = {k: molecules[k] for k in test_data}
train_data = {k: molecules[k] for k in train_data}
#%%
data_train = pd.concat([train_data[k] for k in train_data])
data_test = pd.concat([test_data[k] for k in test_data])
#%%
print(data_train.columns)
#%%
data_train = data_train[['nu', 'sw', 'a', 'gamma_air', 'v', 'J', 'vpp', 'Jpp', 'molecule_weight', 'air_weight', 'molecule_dipole']]
data_test = data_test[['nu', 'sw', 'a', 'gamma_air', 'v', 'J', 'vpp', 'Jpp', 'molecule_weight', 'air_weight', 'molecule_dipole']]
#%%
data_test = data_test.sample(frac=1)
data_train = data_train.sample(frac=1)
#%%
data_test
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#%%
X_train = data_train.drop(['gamma_air'], axis=1)
y_train = data_train['gamma_air']
X_test = data_test.drop(['gamma_air'], axis=1)
y_test = data_test['gamma_air']

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0))
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('score = '+str(pipe.score(X_test, y_test)))

y_test_plot = y_test.to_numpy()

#print(data_dun.columns)


nu_plot = X_test['J'].to_numpy()


yerr = (y_pred - y_test_plot)#/np.std(y_test_plot)


fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 0, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', label="Pressure broadening given by HITRAN")
plt.plot( nu_plot[:200], y_pred[:200], 'bx', label="Predicted pressure broadening")
plt.ylabel(r'Pressure Broadening, $\gamma_{air}$ / $cm^{-1}atm^{-1}$')
plt.legend(loc='upper right')
plt.title('Predicted $\gamma_{Air}$ for lines in $CO$')



#frame2=fig1.add_axes((1, -.35, 2.5, .35))
#plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-', label="error in predicted value")

plt.xlabel('J, rotational quantum number')
#plt.legend(loc='lower right')

plt.show()
#%%
nu_plot = X_test['molecule_weight'].to_numpy()


yerr = (y_pred - y_test_plot)/np.std(y_test_plot)


fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

frame2=fig1.add_axes((1, .1, 2.5, 1))
plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
 
#%%

#%%

#%%
