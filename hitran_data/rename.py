#%%
import pandas as pd
import numpy as np
import os
from glob import iglob

rootdir_glob = '/Users/elizabeth/Desktop/line_broadening/hitran_data/**/*' # Note the added asterisks
# This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) if f[-3:] == "txt" if "readme" not in f]