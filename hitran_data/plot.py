#%%
import pandas as pd
import numpy as np

data = pd.read_csv("CO2/1_iso.csv")
#%%
import matplotlib.pyplot as plt
#%%
#molecules[key] = data.sample(frac=1)
#%%
fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(data['J'][-1000:], data['gamma_air'][-1000:], 'rx', label='co2')
plt.xlabel('Quantum Number J')
plt.ylabel('$\gamma_{air}$, Pressure Broadening by Air')
plt.title('Variation in Pressure Broadening with J for Carbon Monoxide')
#plt.legend()
plt.show()
#%%
from scipy.stats import norm, cauchy
from scipy.special import voigt_profile
#%%
x = np.linspace(-5, 5, 1000)
zo  = np.linspace(0, 0, 1000)
y = norm.pdf(x, 0, 1)

plt.plot(x, y)
plt.plot(x, zo, 'black')

plt.axis('off')

#%%
x = np.linspace(-5, 5, 1000)
y = cauchy.pdf(x, 0, 1)

plt.plot(x, y)
plt.plot(x, zo, 'black')

plt.axis('off')
#%%
x = np.linspace(-5, 5, 1000)
y = voigt_profile(x, 1, 1)

plt.plot(x, y)
plt.plot(x, zo, 'black')

plt.axis('off')
#%%
x = np.linspace(-5, 5, 1000)
zo  = np.linspace(0, 0, 1000)
y = norm.pdf(x, 0, 1)

plt.plot(x, y, label='Gaussian')




y = cauchy.pdf(x, 0, 1)

plt.plot(x, y, label='Lorenzian')


y = voigt_profile(x, 1, 1)

plt.plot(x, y, label='Voigt')
plt.plot(x, zo, 'black')
plt.legend()
plt.axis('off')

plt.savefig('profile.jpeg')

from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi=600)

#%%
