#%%
import pandas as pd
import numpy as np

co = pd.read_csv("CO/1_iso.csv")
o2 = pd.read_csv("O2/O2_1_iso.csv")
no = pd.read_csv("NO/1_iso.csv")
oh = pd.read_csv("OH/1_iso.csv")
hf = pd.read_csv("HF/1_iso.csv")
hcl = pd.read_csv("HCl/1_iso.csv")
hbr = pd.read_csv("HBr/1_iso.csv")
hi = pd.read_csv("HI/1_iso.csv")
clo = pd.read_csv("ClO/1_iso.csv")
n2 = pd.read_csv("N2/N2_1_iso.csv")
h2 = pd.read_csv("H2/H2_1_iso.csv")
cs = pd.read_csv("CS/1_iso.csv")
so = pd.read_csv("SO/1_iso.csv")
#%%
molecules = {"CO": co, "NO": no, "OH": oh,"HF": hf, "HCl": hcl, "HBr": hbr, "HI": hi, "ClO": clo, "CS": cs, "SO": so, "O2": o2, "N2": n2, "H2": h2}
#%%
#print(molecules)

for key, data in molecules.items():
    molecules[key] = data.sample(frac=1)
#%%
import matplotlib.pyplot as plt
#%%
molecules['CO'].columns
#%%
for key, data in molecules.items():
    fig1 = plt.figure(1)
    frame1=fig1.add_axes((1, 1.1, 2.5, 1))
    plt.plot(data['J'][-1000:], data['gamma_air'][-1000:], 'rx', label=key)
    plt.legend()
    plt.show()
#%%
fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 3.5, 3))
for key, data in molecules.items():
    if key == 'O2':
        break

    plt.plot(data['J'][-1000:], data['gamma_air'][-1000:], '.', label=key)
    
lgnd = plt.legend(markerscale=5, fontsize=20)

plt.show()
#%%
