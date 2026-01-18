#%%
import pandas as pd
import numpy as np

df = pd.read_csv("H2O/6278f3b7.txt")
#%%
print(df.columns)
#%%
print(df['gamma_air-err'])
#%%
filtered = df.iloc[:, 0:15]
#%%
filtered.columns
#%%
#filtered2 = df[['nu-err', 'sw-err', 'gamma_air-err', 'n_air-err']]
#%%
#data = filtered.join(filtered2)
#%%
filtered
#%% md
# 
#%%
data = filtered.drop(['global_iso_id', 'molec_id'], axis=1)
#%%
one_hot = pd.get_dummies(data['local_iso_id'])
data = data.drop('local_iso_id', axis = 1)
data_dun = one_hot.join(data)
#%%
data_dun
#%%
new = data_dun["statep"].str.split(";", expand=True)
print(new)
#%%
data_dun["ElecStateLabel"]= new[0].str.replace('ElecStateLabel=', '')
data_dun["v1"]= new[1].str.replace('v1=', '')
data_dun["v2"]= new[2].str.replace('v2=', '')
data_dun["v3"]= new[3].str.replace('v3=', '')
data_dun["J"]= new[4].str.replace('J=', '')
data_dun["Ka"]= new[5].str.replace('Ka=', '')
data_dun["Kc"]= new[6].str.replace('Kc=', '')
# Dropping old Name columns
data_dun.drop(columns =['statep'], inplace = True)
  
# df display
data_dun
#%%
new2 = data_dun["statepp"].str.split(";", expand=True)

data_dun["ElecStateLabelpp"]= new2[0].str.replace('ElecStateLabel=', '')
data_dun["v1pp"]= new2[1].str.replace('v1=', '')
data_dun["v2pp"]= new2[2].str.replace('v2=', '')
data_dun["v3pp"]= new2[3].str.replace('v3=', '')
data_dun["Jpp"]= new2[4].str.replace('J=', '')
data_dun["Kapp"]= new2[5].str.replace('Ka=', '')
data_dun["Kcpp"]= new2[6].str.replace('Kc=', '')
# Dropping old Name columns
data_dun.drop(columns =['statepp'], inplace = True)
  
# df display
data_dun
#%%
data_dun.drop(columns =['ElecStateLabelpp'], inplace = True)
data_dun.drop(columns =['ElecStateLabel'], inplace = True)
#%%
data_dun
#%%
X = data_dun.drop(['gamma_air', 'n_air', 'gamma_self', 'n_self'], axis=1)
y = data_dun['gamma_air']
y2 = data_dun['n_air']
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#%%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0))
pipe.fit(X_train, y_train)
#%%
y_pred = pipe.predict(X_test)
#%%
pipe.score(X_test, y_test)
#%%
print(y_pred, y_test)
#%%
y_test_plot = y_test.to_numpy()
#%%
print(X_test['nu'])
#%%
data_dun.columns
#%%
nu_plot = X_test['elower'].to_numpy()
#%%
import matplotlib.pyplot as plt

plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')
plt.show()
#%%
yerr = (y_pred - y_test_plot)/np.std(y_test_plot)
#%%
fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

frame2=fig1.add_axes((1, .1, 2.5, 1))
plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
 
#%%
X = data_dun.drop(['gamma_air', 'n_air', 'gamma_self', 'n_self'], axis=1)
y = data_dun['gamma_self']
y2 = data_dun['n_self']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0))
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print(pipe.score(X_test, y_test))

y_test_plot = y_test.to_numpy()

nu_plot = X_test['elower'].to_numpy()

yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

frame2=fig1.add_axes((1, .1, 2.5, 1))
plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
#%%
X = data_dun.drop(['gamma_air', 'n_air', 'gamma_self', 'n_self'], axis=1)
y = data_dun['gamma_self']
y2 = data_dun['n_self']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0))
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print(pipe.score(X_test, y_test))

y_test_plot = y_test.to_numpy()

nu_plot = X_test['J'].to_numpy()

yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

frame2=fig1.add_axes((1, .1, 2.5, 1))
plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
#%%
X = data_dun.drop(['gamma_air', 'n_air', 'gamma_self', 'n_self'], axis=1)
y = data_dun['gamma_self']
y2 = data_dun['n_self']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0))
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print(pipe.score(X_test, y_test))

y_test_plot = y_test.to_numpy()

nu_plot = X_test['Ka'].to_numpy()

yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

frame2=fig1.add_axes((1, .1, 2.5, 1))
plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
#%%
X = data_dun.drop(['gamma_air', 'n_air', 'gamma_self', 'n_self'], axis=1)
y = data_dun['gamma_self']
y2 = data_dun['n_self']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0))
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print(pipe.score(X_test, y_test))

y_test_plot = y_test.to_numpy()

nu_plot = X_test['Kc'].to_numpy()

yerr = (y_pred - y_test_plot)/np.std(y_test_plot)

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', nu_plot[:200], y_pred[:200], 'bx')

frame2=fig1.add_axes((1, .1, 2.5, 1))
plt.plot(nu_plot[:200], yerr[:200], 'gx', nu_plot[:200], np.zeros(200), 'k-')
#%%

#%%

#%%

#%%

#%%
