#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
X = np.linspace(1, 30, 30).reshape(-1, 1)
#%%
y = X*3 + 1 + np.random.rand(30, 1)*30
#%%
print(X)
#%%
print(y)
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
#%%

#%%

X_train, X_test, y_train, y_test = train_test_split(X, y)



pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('score = '+str(pipe.score(X_test, y_test)))

y_test_plot = y_test


nu_plot = X_test


yerr = (y_pred - y_test_plot)/np.std(y_test_plot)


#%%
from matplotlib.pyplot import figure

figure(figsize=((2, 3.5)), dpi=500)

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(nu_plot[:200], y_test_plot[:200], 'rx', markersize=12, label='Test data points')
plt.plot(nu_plot[:200], y_pred[:200], 'b-', linewidth=1.5, label=r'Predicted fit to $Y = aX + b$')
plt.xlim([0, 30])
plt.ylim([0, 120])

plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper left')

plt.show()
#%%

#%%
from scipy import special as sp
#%%
X_voigt = np.linspace(-20, 20, 5000).reshape(-1, 1)
#%%
y_voigt = 0.25 - sp.voigt_profile(X_voigt, 1, 1)
#%%

figure(figsize=((2, 3.5)), dpi=500)

fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(X_voigt, y_voigt, 'k', linewidth=1.4, label='Test data points')
#plt.plot(nu_plot[:200], y_pred[:200], 'b-', linewidth=1.5, label=r'Predicted fit to $Y = aX + b$')
#plt.xlim([0, 30])
#plt.ylim([0, 120])

plt.xlabel('X')
plt.ylabel('Voigt profile')
#plt.legend(loc='upper left')

#%%

#%%
