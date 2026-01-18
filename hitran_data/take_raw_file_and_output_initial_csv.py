import pandas as pd
import numpy as np
import io

# ---------------------------------------------------------
# 1. Setup & Load Data
# ---------------------------------------------------------
# Replace this with your actual file path:
file_path = "New_hitran_2025/6926f634.txt"

# Read the data
df = pd.read_csv(file_path)

# Filter for the most abundant isotopologue
df_main = df.loc[df['local_iso_id'] == 1].copy()


# ---------------------------------------------------------
# 2. Generic Parsing Function
# ---------------------------------------------------------
def parse_quantum_numbers(df, source_col, suffix=""):
    """
    Parses a column of 'key=val;key2=val2;' strings into separate columns.
    """
    # Ensure string and append delimiter to catch last value
    series_to_parse = df[source_col].astype(str) + ';'

    # Regex to capture Key=Value pairs
    pattern = r'(?P<key>[^=;]+)=(?P<value>[^;]*)'

    # Extract all matches (Creates MultiIndex)
    extracted = series_to_parse.str.extractall(pattern)

    # Reset index to allow pivoting
    extracted = extracted.reset_index(level=1, drop=True).reset_index()

    # Pivot: Index -> Row ID, Columns -> Keys (v1, J, etc), Values -> Parsed Values
    pivoted = extracted.pivot(index='index', columns='key', values='value')

    # Rename columns with suffix (e.g. 'J' -> 'Jpp')
    if suffix:
        pivoted.columns = [f"{col}{suffix}" for col in pivoted.columns]

    return pivoted


# ---------------------------------------------------------
# 3. Execute Parsing & Merge
# ---------------------------------------------------------
print("Parsing Upper State...")
df_upper = parse_quantum_numbers(df_main, 'statep', suffix="")

print("Parsing Lower State...")
df_lower = parse_quantum_numbers(df_main, 'statepp', suffix="pp")

# Join back to main dataframe
df_final = df_main.join(df_upper).join(df_lower)

# Remove the original raw string columns
df_final.drop(columns=['statep', 'statepp'], inplace=True)

# Convert to numeric where possible (leaves strings like 'A1' alone)
df_final = df_final.apply(pd.to_numeric, errors='ignore')

# ---------------------------------------------------------
# 4. Split by Molecule, Clean, and Save
# ---------------------------------------------------------

# Mapping for filenames
hitran_molecule_map = {
    1: 'H2O', 2: 'CO2', 3: 'O3', 4: 'N2O', 5: 'CO',
    6: 'CH4', 7: 'O2', 8: 'NO', 9: 'SO2', 10: 'NO2',
    11: 'NH3', 12: 'HNO3', 13: 'OH', 14: 'HF', 15: 'HCl',
    16: 'HBr', 17: 'HI', 18: 'ClO', 19: 'OCS', 20: 'H2CO',
    21: 'HOCl', 22: 'N2', 23: 'HCN', 24: 'CH3Cl', 25: 'H2O2',
    26: 'C2H2', 27: 'C2H6', 28: 'PH3', 29: 'COF2', 30: 'SF6',
    31: 'H2S', 32: 'HCOOH', 33: 'HO2', 34: 'O', 35: 'ClONO2',
    36: 'NO+', 37: 'HOBr', 38: 'C2H4', 39: 'CH3OH', 40: 'CH3Br',
    41: 'CH3CN', 42: 'CF4', 43: 'C4H2', 44: 'HC3N', 45: 'H2',
    46: 'CS', 47: 'SO3', 48: 'C2N2', 49: 'COCl2', 50: 'SO',
    51: 'CH3F', 52: 'GeH4', 53: 'CS2', 54: 'CH3I', 55: 'NF3'
}

print(f"Splitting data by Molec_ID and cleaning NaN columns...")

for mol_id, group_df in df_final.groupby('molec_id'):
    # 1. Drop columns that are entirely NaN for this specific molecule
    #    This removes 'v3' columns for diatomic molecules, etc.
    cleaned_df = group_df.dropna(axis=1, how='all')

    # 2. Get Molecule Name
    mol_name = hitran_molecule_map.get(mol_id, f"Molecule_{mol_id}")

    # 3. Construct Filename
    filename = f"New_hitran_2025/{mol_name}_1_iso.csv"

    # 4. Save
    cleaned_df.to_csv(filename, index=False)
    print(f"Saved {mol_name} (ID: {mol_id}): Dropped {len(group_df.columns) - len(cleaned_df.columns)} empty columns.")