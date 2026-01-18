import pandas as pd
import numpy as np
import os
from glob import iglob

rootdir_glob = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/hitran_data/**/*' # Note the added asterisks
# This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-10:] == "/1_iso.csv" if "readme" not in f]


db = {}

for f in file_list:
    i = f[76:-10]
    db[i] = pd.read_csv(f)


for key, data in db.items():
    print(key)


for key, data in db.items():
    # symmetric linear
    if key in ['C2H2', 'CO2', 'H2', 'N2', 'O2']:
        data['symmetry'] = 1
        db[key] = data

    # asymmetric linear
    if key in ['ClO', 'CO', 'CS', 'HBr', 'HCl', 'HF', 'N2O', 'NO', 'OCS', 'OH']:
        data['symmetry'] = 4
        db[key] = data

    # symmetric top
    if key in ['C2H6', 'CH3F', 'NH3']:
        data['symmetry'] = 3
        db[key] = data

    # asymmetric top
    if key in ['C2H4', 'CH3OH', 'COF2', 'H2CO', 'H2O2', 'H2O', 'HO2', 'NO2', 'O3', 'SO2']:
        data['symmetry'] = 2
        db[key] = data

    # spherical top
    if key in ['CH4']:
        data['symmetry'] = 5
        db[key] = data


for key, data in db.items():
    data.to_csv(f"{key}/1_iso.csv")
