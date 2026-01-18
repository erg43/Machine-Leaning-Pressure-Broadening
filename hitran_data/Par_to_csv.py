#%%
import pandas as pd
import numpy as np



df = pd.read_csv("CH3CN/68c163c5.par", header=None)
#%% md
# 
#%%
import pandas as pd
import numpy as np
import os
from glob import iglob


pd.read_csv('NO/NO_1_iso.csv')
#%%
dr = pd.read_csv("GeH4/raw.txt")
print(dr.columns)
#%%
df['molec_id'] = [x[:2] for x in df[0]]
df['local_iso_id'] = [x[2:3] for x in df[0]]
df['nu'] = [x[3:15] for x in df[0]]
df['sw'] = [x[15:25] for x in df[0]]
df['a'] = [x[25:35] for x in df[0]]
df['gamma_air'] = [x[35:40] for x in df[0]]
df['gamma_self'] = [x[40:45] for x in df[0]]
df['elower'] = [x[45:55] for x in df[0]]
df['n_air'] = [x[55:59] for x in df[0]]
df['delta_air'] = [x[59:67] for x in df[0]]
df['statep'] = [x[67:82] for x in df[0]]
df['statepp'] = [x[82:97] for x in df[0]]
df['JKp'] = [x[97:112] for x in df[0]]
df['JKpp'] = [x[112:127] for x in df[0]]
df['uncertainties'] = [x[127:133] for x in df[0]]
df['references'] = [x[133:145] for x in df[0]]
df['flag'] = [x[145:146] for x in df[0]]
df['gp'] = [x[146:153] for x in df[0]]
df['gpp'] = [x[153:160] for x in df[0]]
#%%
df.drop([0], axis=1)
#%%
df['gamma_air-err'] = [x[2] for x in df['uncertainties']]
df['gamma_self-err'] = [x[2] for x in df['uncertainties']]
#%%
df['gamma_air-err'].value_counts()
#%%
df['local_iso_id'].value_counts()
#print(len(df['local_iso_id']))
#print(int(df['local_iso_id'])==1)
#df_main = df.loc[df['local_iso_id'] == 1]
#print(len(df_main['local_iso_id']))
#%%
df_main = df
#%%

#%%
#df_main["ElecStateLabel"]= df_main['statep'].str.extract(r'(ElecStateLabel=)(.+?);')[1]
#df_main["v1"]= df_main['statep'].str.extract(r'(v1=)(.+?);')[1]
#df_main["Lambda"]= df_main['statep'].str.extract(r'(Lambda=)(.+?);')[1]
#df_main["Omega"]= df_main['statep'].str.extract(r'(Omega=)(.+?);')[1]
#df_main["v"]= df_main['statep'].str.extract(r'(v=)(.+?);')[1]
#df_main["S"]= df_main['statep'].str.extract(r'(S=)(.+?);')[1]
#df_main["v2"]= df_main['statep'].str.extract(r'(v2=)(.+?);')[1]
#df_main["v3"]= df_main['statep'].str.extract(r'(v3=)(.+?);')[1]
#df_main["l2"]= df_main['statep'].str.extract(r'(l2=)(.+?);')[1]
#df_main["v5"]= df_main['statep'].str.extract(r'(v5=)(.+?);')[1]
#df_main["v4"]= df_main['statep'].str.extract(r'(v4=)(.+?);')[1]
#df_main["v6"]= df_main['statep'].str.extract(r'(v6=)(.+?);')[1]
#df_main["v7"]= df_main['statep'].str.extract(r'(v7=)(.+?);')[1]
#df_main["v8"]= df_main['statep'].str.extract(r'(v8=)(.+?);')[1]

#df_main["v9"]= df_main['statep'].str.extract(r'(v9=)(.+?);')[1]
#df_main["v10"]= df_main['statep'].str.extract(r'(v10=)(.+?);')[1]
#df_main["v11"]= df_main['statep'].str.extract(r'(v11=)(.+?);')[1]
#df_main["v12"]= df_main['statep'].str.extract(r'(v12=)(.+?);')[1]
#df_main["l3"]= df_main['statep'].str.extract(r'(l3=)(.+?);')[1]
#df_main["l4"]= df_main['statep'].str.extract(r'(l4=)(.+?);')[1]
#df_main["l5"]= df_main['statep'].str.extract(r'(l5=)(.+?);')[1]
#df_main["l"]= df_main['statep'].str.extract(r'(;l=)(.+?);')[1]
#df_main["l6"]= df_main['statep'].str.extract(r'(;l6=)(.+?);')[1]
#df_main["l7"]= df_main['statep'].str.extract(r'(;l7=)(.+?);')[1]
#df_main["l8"]= df_main['statep'].str.extract(r'(;l8=)(.+?);')[1]
#df_main["l9"]= df_main['statep'].str.extract(r'(;l9=)(.+?);')[1]
#df_main["vibinv"]= df_main['statep'].str.extract(r'(vibInv=)(.+?);')[1]
#df_main["vibrefl"]= df_main['statep'].str.extract(r'(vibRefl=)(.+?);')[1]
#df_main["vibsym"]= df_main['statep'].str.extract(r'(vibSym=)(.+?);')[1]
#df_main["J"]= df_main['statep'].str.extract(r'(J=)(.+?);')[1]
#df_main["Fnuc"]= df_main['statep'].str.extract(r'(F#nuclearSpinRef:I1=)(.+?);')[1]

#df_main["Ka"]= df_main['statep'].str.extract(r'(Ka=)(.+?);')[1]
#df_main["Kc"]= df_main['statep'].str.extract(r'(Kc=)(.+?);')[1]
#df_main["F"]= df_main['statep'].str.extract(r'(F=)(.+?);')[1]
#df_main["K"]= df_main['statep'].str.extract(r'(K=)(.+?);')[1]
#df_main["rotsym"]= df_main['statep'].str.extract(r'(rotSym=)(.+?);')[1]
#df_main["rovibsym"]= df_main['statep'].str.extract(r'(rovibSym=)(.+?);')[1]
#df_main["r"]= df_main['statep'].str.extract(r'(r=)(.+?);')[1]
#df_main["parity"]= df_main['statep'].str.extract(r'(parity=)(.+?);')[1]
#df_main["kronigparity"]= df_main['statep'].str.extract(r'(kronigParity=)(.+?);')[1]
#df_main["N"]= df_main['statep'].str.extract(r'(N=)(.+?);')[1]
#df_main["n"]= df_main['statep'].str.extract(r'(n=)(.+?);')[1]
#df_main["alpha"]= df_main['statep'].str.extract(r'(alpha=)(.+?);')[1]
#df_main["tau"]= new[10].str.replace('tau=', '')
# Dropping old Name columns
#df_main.drop(columns =['statep'], inplace = True)
  
# df display
#print(df_main)
#%%
#df_main['statepp'] = df_main["statepp"]+str(';')
#%%
#new = df_main["statepp"].str.split(";", expand=True)
#print(new.iloc[0])
#print(new.iloc[-1])
#%%
#new[9].unique()
#%%
#df_main["ElecStateLabelpp"]= new[0].str.replace('ElecStateLabel=', '')
#df_main["v1pp"]= df_main['statepp'].str.extract(r'(v1=)(.+?);')[1]
#df_main["Lambdapp"]= df_main['statepp'].str.extract(r'(Lambda=)(.+?);')[1]
#df_main["Omegapp"]= df_main['statepp'].str.extract(r'(Omega=)(.+?);')[1]
#df_main["Spp"]= df_main['statepp'].str.extract(r'(S=)(.+?);')[1]
#df_main["vpp"]= df_main['statepp'].str.extract(r'(v=)(.+?);')[1]


#df_main["v2pp"]= df_main['statepp'].str.extract(r'(v2=)(.+?);')[1]
#df_main["l2pp"]= df_main['statepp'].str.extract(r'(l2=)(.+?);')[1]
#df_main["v3pp"]= df_main['statepp'].str.extract(r'(v3=)(.+?);')[1]
#df_main["v4pp"]= df_main['statepp'].str.extract(r'(v4=)(.+?);')[1]
#df_main["v5pp"]= df_main['statepp'].str.extract(r'(v5=)(.+?);')[1]
#df_main["v6pp"]= df_main['statepp'].str.extract(r'(v6=)(.+?);')[1]
#df_main["v7pp"]= df_main['statepp'].str.extract(r'(v7=)(.+?);')[1]
#df_main["v8pp"]= df_main['statepp'].str.extract(r'(v8=)(.+?);')[1]
#df_main["v9pp"]= df_main['statepp'].str.extract(r'(v9=)(.+?);')[1]
#df_main["v10pp"]= df_main['statepp'].str.extract(r'(v10=)(.+?);')[1]
#df_main["v12pp"]= df_main['statepp'].str.extract(r'(v12=)(.+?);')[1]
#df_main["l3pp"]= df_main['statepp'].str.extract(r'(l3=)(.+?);')[1]
#df_main["l4pp"]= df_main['statepp'].str.extract(r'(l4=)(.+?);')[1]
#df_main["l5pp"]= df_main['statepp'].str.extract(r'(l5=)(.+?);')[1]
#df_main["lpp"]= df_main['statepp'].str.extract(r'(;l=)(.+?);')[1]
#df_main["l6pp"]= df_main['statepp'].str.extract(r'(;l6=)(.+?);')[1]
#df_main["l7pp"]= df_main['statepp'].str.extract(r'(;l7=)(.+?);')[1]
#df_main["l8pp"]= df_main['statepp'].str.extract(r'(;l8=)(.+?);')[1]
#df_main["l9pp"]= df_main['statepp'].str.extract(r'(;l9=)(.+?);')[1]

#df_main["vibinvpp"]= df_main['statepp'].str.extract(r'(vibInv=)(.+?);')[1]
#df_main["vibreflpp"]= df_main['statepp'].str.extract(r'(vibRefl=)(.+?);')[1]
#df_main["vibsympp"]= df_main['statepp'].str.extract(r'(vibSym=)(.+?);')[1]
#df_main["Jpp"]= df_main['statepp'].str.extract(r'(J=)(.+?);')[1]
#df_main["Fnucpp"]= df_main['statepp'].str.extract(r'(F#nuclearSpinRef:I1=)(.+?);')[1]

#df_main["Kapp"]= df_main['statepp'].str.extract(r'(Ka=)(.+?);')[1]
#df_main["Kcpp"]= df_main['statepp'].str.extract(r'(Kc=)(.+?);')[1]
#df_main["Fpp"]= df_main['statepp'].str.extract(r'(F=)(.+?);')[1]
#df_main["Kpp"]= df_main['statepp'].str.extract(r'(K=)(.+?);')[1]
#df_main["rotsympp"]= df_main['statepp'].str.extract(r'(rotSym=)(.+?);')[1]
#df_main["rovibsympp"]= df_main['statepp'].str.extract(r'(rovibSym=)(.+?);')[1]
#df_main["rpp"]= df_main['statepp'].str.extract(r'(r=)(.+?);')[1]
#df_main["paritypp"]= df_main['statepp'].str.extract(r'(parity=)(.+?);')[1]
#df_main["kronigparitypp"]= df_main['statepp'].str.extract(r'(kronigParity=)(.+?);')[1]
#df_main["Npp"]= df_main['statepp'].str.extract(r'(N=)(.+?);')[1]
#df_main["npp"]= df_main['statepp'].str.extract(r'(n=)(.+?);')[1]
#df_main["alphapp"]= df_main['statepp'].str.extract(r'(alpha=)(.+?);')[1]
#df_main["taupp"]= new[10].str.replace('tau=', '')

# Dropping old Name columns
#df_main.drop(columns =['statepp'], inplace = True)
  
# df display
#df_main
#%%
print(df_main.iloc[0][-20:])
#%%
df_main['J'] = df_main['JKp'].str[:3]
df_main['Jpp'] = df_main['JKpp'].str[:3]
print(max(df_main['Jpp']))
#df_main = df_main.drop(['K', 'Kpp'], axis=1)
df_main['K'] = df_main['JKp'].str[3:6]
df_main['Kpp'] = df_main['JKpp'].str[3:6]
#df_main['Kc'] = df_main['JKp'].str[6:9]
#df_main['Kcpp'] = df_main['JKpp'].str[6:9]
#print(max(df_main['Ka']))
print(df_main)
#%%
#df_main = df_main[['J', 'Jpp', 'Ka', 'Kc', 'Kapp', 'Kcpp']]
#%%
df_main = df_main.sample(frac=1)
df_main = df_main.apply(pd.to_numeric, errors='ignore')
#%%
import matplotlib.pyplot as plt


fig1 = plt.figure(1)
frame1=fig1.add_axes((1, 1.1, 2.5, 1))
plt.plot(df_main['J'][:1000], df_main['gamma_air'][:1000], '.')
plt.show()
#%%
df_main.to_csv("S2/S2_1_iso_test.csv")
#%%
    
#%%

#%%

#%%

#%%
missing C2H2!!!
missing C2H6!!!
missing PH3!!!   l



CH3Cl    Fnuc
OCS      parity, kronig parity
ClO      Fnuc
OH       Fnuc
HNO3     F_DUPFD_CLOEXEC
NH3      all
NO       Fnuc
CH4      rovibsym, alpha








for i, row in new[9].items():
    print(i, row)
    if row[0] == 'v':
        df_main["vibreflpp"].iloc[i]= row.replace('vibRefl=', '')
        df_main["Jpp"].iloc[i]= new[10].iloc[i].replace('J=', '')
        df_main["paritypp"].iloc[i]= new[11].iloc[i].replace('parity=', '')
        df_main["kronigparitypp"].iloc[i]= new[12].iloc[i].replace('kronigParity=', '')

    else:




#%%

#%%
