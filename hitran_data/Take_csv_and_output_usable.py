import pandas as pd
import numpy as np
import os
from pathlib import Path

# =============================================================================
# 1. RUN CONFIGURATION (User Flags)
# =============================================================================

# Set this to:
#   1. "ALL"            -> To run every molecule found.
#   2. "CO"             -> To run a single molecule.
#   3. ["CO", "H2O"]    -> To run a specific list of molecules.
TARGET_MOLECULES = ['NO2', 'CS2', 'CH3F', 'HCN', 'HNO3', 'H2CO']

# Root directory for HITRAN data
ROOT_DIR = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/hitran_data/'

# =============================================================================
# 2. GLOBAL CONSTANTS
# =============================================================================

CONSTANTS = {
    'air_weight': 28.97,
    'He_weight': 4.002602,
    'H2_weight': 2.01588,
    'CO2_weight': 44.0095,
    'H2O_weight': 18.0153,
    'air_quadrupole_z': -1.15552,
    'air_quadrupole_x': 0.57776,
    'H2_quadrupole_z': 0.520,
    'H2_quadrupole_x': -0.260,
    'CO2_quadrupole_z': -4.278,
    'CO2_quadrupole_x': 2.139,
    'H2O_dipole': 1.857
}

# =============================================================================
# 3. MOLECULAR DATABASE
# =============================================================================
# Symmetry: 1=Linear, 2=Asym Top, 3=Sym Top, 4=Asym Linear, 5=Spherical Top

MOLECULAR_DATA = {
    'H2O': {'dipole': 1.857, 'polar': 1.501, 'B0a': 27.8806, 'B0b': 14.5218, 'B0c': 9.2777, 'd': 285.5, 'symmetry': 2},
    'CH3F': {'dipole': 1.847, 'polar': 2.540, 'B0a': 5.1820, 'B0b': 0.85179, 'B0c': 0.85179, 'd': 373, 'symmetry': 3},
    'C2H6': {'dipole': 0, 'polar': 4.226, 'B0a': 2.5197, 'B0b': 0.68341, 'B0c': 0.68341, 'd': 356.5, 'symmetry': 3},
    'H2': {'dipole': 0, 'polar': 0.787, 'B0a': 100000, 'B0b': 60.853, 'B0c': 60.853, 'd': 289, 'symmetry': 1},
    'HO2': {'dipole': 2.090, 'polar': 2.027, 'B0a': 20.3565, 'B0b': 1.1179, 'B0c': 1.0565, 'd': 311, 'symmetry': 2},
    'NO': {'dipole': 0.159, 'polar': 1.698, 'B0a': 100000, 'B0b': 1.6961, 'B0c': 1.6961, 'd': 317, 'symmetry': 4},
    'O2': {'dipole': 0, 'polar': 1.562, 'B0a': 100000, 'B0b': 1.4377, 'B0c': 1.4377, 'd': 346, 'symmetry': 1},
    'SO2': {'dipole': 1.633, 'polar': 3.882, 'B0a': 2.0274, 'B0b': 0.3442, 'B0c': 0.2935, 'd': 411.2, 'symmetry': 2},
    'CO': {'dipole': 0.110, 'polar': 1.953, 'B0a': 100000, 'B0b': 1.9225, 'B0c': 1.9225, 'd': 376, 'symmetry': 4},
    'CS': {'dipole': 1.957, 'polar': 5.397, 'B0a': 100000, 'B0b': 0.8171, 'B0c': 0.8171, 'd': 358, 'symmetry': 4},
    'CH3OH': {'dipole': 1.672, 'polar': 3.210, 'B0a': 4.2524, 'B0b': 0.8232, 'B0c': 0.7929, 'd': 329, 'symmetry': 2},
    'O3': {'dipole': 0.534, 'polar': 3.079, 'B0a': 3.5537, 'B0b': 0.4453, 'B0c': 0.3948, 'd': 400, 'symmetry': 2},
    'ClO': {'dipole': 1.297, 'polar': 3.02, 'B0a': 100000, 'B0b': 0.6205, 'B0c': 0.6205, 'd': 384.2, 'symmetry': 4},
    'OCS': {'dipole': 0.715, 'polar': 5.090, 'B0a': 100000, 'B0b': 0.2029, 'B0c': 0.2029, 'd': 413, 'symmetry': 4},
    'CH4': {'dipole': 0, 'polar': 2.448, 'B0a': 5.2412, 'B0b': 5.2412, 'B0c': 5.2412, 'd': 380, 'symmetry': 5},
    'C2H4': {'dipole': 0, 'polar': 4.188, 'B0a': 4.8280, 'B0b': 1.0012, 'B0c': 0.8282, 'd': 390, 'symmetry': 2},
    'H2O2': {'dipole': 1.572, 'polar': 1.121, 'B0a': 10.069, 'B0b': 0.8743, 'B0c': 0.8372, 'd': 314, 'symmetry': 2},
    'COF2': {'dipole': 0.951, 'polar': 2.8, 'B0a': 0.3941, 'B0b': 0.3920, 'B0c': 0.1962, 'd': 400, 'symmetry': 2},
    'N2O': {'dipole': 0.161, 'polar': 2.998, 'B0a': 100000, 'B0b': 0.4190, 'B0c': 0.4190, 'd': 383, 'symmetry': 4},
    'H2CO': {'dipole': 2.331, 'polar': 2.770, 'B0a': 9.4055, 'B0b': 1.2954, 'B0c': 1.1343, 'd': 321.5, 'symmetry': 2},
    'C2H2': {'dipole': 0, 'polar': 3.487, 'B0a': 100000, 'B0b': 1.1766, 'B0c': 1.1766, 'd': 330, 'symmetry': 1},
    'HCl': {'dipole': 1.109, 'polar': 2.515, 'B0a': 100000, 'B0b': 10.4402, 'B0c': 10.4402, 'd': 333.9, 'symmetry': 4},
    'NO2': {'dipole': 0.316, 'polar': 2.910, 'B0a': 8.0023, 'B0b': 0.4337, 'B0c': 0.4105, 'd': 346, 'symmetry': 2},
    'HF': {'dipole': 1.827, 'polar': 0.800, 'B0a': 100000, 'B0b': 20.5597, 'B0c': 20.5597, 'd': 314.8, 'symmetry': 4},
    'N2': {'dipole': 0, 'polar': 1.710, 'B0a': 100000, 'B0b': 1.6720, 'B0c': 1.6720, 'd': 364, 'symmetry': 1},
    'HBr': {'dipole': 0.828, 'polar': 3.616, 'B0a': 100000, 'B0b': 8.3492, 'B0c': 8.3492, 'd': 335.3, 'symmetry': 4},
    'OH': {'dipole': 1.655, 'polar': 0.466, 'B0a': 100000, 'B0b': 18.552, 'B0c': 18.552, 'd': 336, 'symmetry': 4},
    'NH3': {'dipole': 1.472, 'polar': 2.103, 'B0a': 9.9466, 'B0b': 9.9466, 'B0c': 6.2275, 'd': 375, 'symmetry': 3},
    'CO2': {'dipole': 0, 'polar': 2.507, 'B0a': 100000, 'B0b': 0.3902, 'B0c': 0.3902, 'd': 330, 'symmetry': 1},
    'HONO': {'dipole': 1.813, 'polar': 2.247, 'B0a': 3.0986, 'B0b': 0.4178, 'B0c': 0.3675, 'd': 346, 'symmetry': 2},
    'SO3': {'dipole': 0, 'polar': 19.029, 'B0a': 0.346, 'B0b': 0.346, 'B0c': 0.173, 'd': 470, 'symmetry': 3},
    'NF3': {'dipole': 0.235, 'polar': 2.810, 'B0a': 0.3563, 'B0b': 0.3563, 'B0c': 0.1949, 'd': 415.4, 'symmetry': 3},
    'SF6': {'dipole': 0, 'polar': 4.490, 'B0a': 0.0911, 'B0b': 0.0911, 'B0c': 0.0911, 'd': 512.8, 'symmetry': 5},
    'CF4': {'dipole': 0, 'polar': 2.824, 'B0a': 0.1924, 'B0b': 0.1924, 'B0c': 0.1924, 'd': 470, 'symmetry': 5},
    'ClONO2': {'dipole': 0.772, 'polar': 4.509, 'B0a': 0.4038, 'B0b': 0.0926, 'B0c': 0.0753, 'd': 520, 'symmetry': 2},
    'GeH4': {'dipole': 0, 'polar': 4.770, 'B0a': 2.626, 'B0b': 2.626, 'B0c': 2.626, 'd': 430, 'symmetry': 5},
    'COCl2': {'dipole': 0, 'polar': 6.790, 'B0a': 0.264, 'B0b': 0.116, 'B0c': 0.080, 'd': 420, 'symmetry': 2},
    'H2S': {'dipole': 0.97, 'polar': 3.631, 'B0a': 10.347, 'B0b': 9.036, 'B0c': 4.727, 'd': 360, 'symmetry': 2},
    'HCOOH': {'dipole': 1.41, 'polar': 3.319, 'B0a': 2.585, 'B0b': 0.402, 'B0c': 0.347, 'd': 400, 'symmetry': 2},
    'HCN': {'dipole': 2.98, 'polar': 2.593, 'B0a': 100000, 'B0b': 1.478, 'B0c': 1.478, 'd': 363, 'symmetry': 4},
    'CH3Cl': {'dipole': 1.87, 'polar': 4.416, 'B0a': 5.198, 'B0b': 0.443, 'B0c': 0.443, 'd': 425, 'symmetry': 3},
    'PH3': {'dipole': 0.58, 'polar': 4.237, 'B0a': 4.452, 'B0b': 4.452, 'B0c': 3.930, 'd': 450, 'symmetry': 3},
    'HOCl': {'dipole': 1.51, 'polar': 3.438, 'B0a': 20.46, 'B0b': 0.504, 'B0c': 0.491, 'd': 320, 'symmetry': 2},
    'SO': {'dipole': 1.55, 'polar': 4.238, 'B0a': 100000, 'B0b': 0.721, 'B0c': 0.721, 'd': 400, 'symmetry': 4},
    'HC3N': {'dipole': 3.73, 'polar': 4.468, 'B0a': 100000, 'B0b': 0.152, 'B0c': 0.152, 'd': 465, 'symmetry': 4},
    'CS2': {'dipole': 0, 'polar': 8.749, 'B0a': 100000, 'B0b': 0.109, 'B0c': 0.109, 'd': 460, 'symmetry': 1},
    'HI': {'dipole': 0.44, 'polar': 5.453, 'B0a': 100000, 'B0b': 6.426, 'B0c': 6.426, 'd': 400, 'symmetry': 4},
    'CH3CN': {'dipole': 3.92, 'polar': 4.280, 'B0a': 5.280, 'B0b': 0.307, 'B0c': 0.307, 'd': 458, 'symmetry': 3},
    'CH3I': {'dipole': 1.62, 'polar': 7.325, 'B0a': 5.173, 'B0b': 0.250, 'B0c': 0.250, 'd': 550, 'symmetry': 3},
    'CH3Br': {'dipole': 1.81, 'polar': 5.610, 'B0a': 5.246, 'B0b': 0.322, 'B0c': 0.322, 'd': 426, 'symmetry': 3},
    'HNO3': {'dipole': 2.17, 'polar': 3.160, 'B0a': 0.434, 'B0b': 0.404, 'B0c': 0.209, 'd': 480, 'symmetry': 2},
    'C4H2': {'dipole': 0, 'polar': 6.811, 'B0a': 100000, 'B0b': 0.146, 'B0c': 0.146, 'd': 490, 'symmetry': 1},
    'HOBr': {'dipole': 1.38, 'polar': 2.215, 'B0a': 20.47, 'B0b': 0.353, 'B0c': 0.346, 'd': 300, 'symmetry': 2},
    'C2N2': {'dipole': 0, 'polar': 5.015, 'B0a': 100000, 'B0b': 0.157, 'B0c': 0.157, 'd': 440, 'symmetry': 1},
    'S2': {'dipole': 0, 'polar': 4.0, 'B0a': 100000, 'B0b': 0.295, 'B0c': 0.295, 'd': 360, 'symmetry': 1},
}


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def apply_constants_to_df(df, const_dict):
    """Applies global constants (Air weights, Quadrupoles) to every row."""
    for key, val in const_dict.items():
        df[key] = val
    return df


def approximate_quantum_numbers(df, symmetry_type):
    """
    Approximates Ka/Kc columns based on symmetry type.
    """
    # Initialize defaults
    df['Ka_aprox'] = np.nan
    df['Kc_aprox'] = np.nan
    df['Kapp_aprox'] = np.nan
    df['Kcpp_aprox'] = np.nan

    # 1. Symmetric Linear & 4. Asymmetric Linear
    if symmetry_type == 1 or symmetry_type == 4:
        df['Ka_aprox'] = 0
        df['Kc_aprox'] = df['J']
        df['Kapp_aprox'] = 0
        df['Kcpp_aprox'] = df['Jpp']

    # 2. Asymmetric Top
    if 'Ka' in df.columns:
        df['Ka_aprox'] = df['Ka']
        df['Kc_aprox'] = df['Kc']
    if 'Kapp' in df.columns:
        df['Kapp_aprox'] = df['Kapp']
        df['Kcpp_aprox'] = df['Kcpp']

    # 3. Symmetric Top (Approximate with K if available)
    if 'K' in df.columns:
        df['Ka_aprox'] = df['K']
        df['Kapp_aprox'] = df['Kpp']
        # Rough Approx for Kc in prolate limit J - K
        df['Kc_aprox'] = df['J'] - df['K']
        df['Kcpp_aprox'] = df['Jpp'] - df['Kpp']

    # 5. Spherical Top
    if symmetry_type == 5:
        df['Ka_aprox'] = df['J'] / 2
        df['Kapp_aprox'] = df['Jpp'] / 2
        df['Kc_aprox'] = df['J'] / 2
        df['Kcpp_aprox'] = df['Jpp'] / 2

    return df


def determine_multipole_m(df):
    """Determines m value based on columns present."""
    if 'molecule_dipole' in df.columns and (df['molecule_dipole'] != 0).any():
        return 6
    elif 'molecule_quadrupole_x' in df.columns or 'molecule_quadrupole' in df.columns:
        return 8
    return 8


def should_process_molecule(mol_name):
    """Checks the user flag to see if we should run this molecule."""
    if TARGET_MOLECULES == "ALL":
        return True
    elif isinstance(TARGET_MOLECULES, str):
        return mol_name == TARGET_MOLECULES
    elif isinstance(TARGET_MOLECULES, list):
        return mol_name in TARGET_MOLECULES
    return False


# =============================================================================
# 5. MAIN PROCESSING LOGIC
# =============================================================================

def main():
    print(f"Run Mode: {TARGET_MOLECULES}")
    print(f"Searching for molecules in {ROOT_DIR}...")

    # Looking for files specifically matching "MoleculeName_1_iso.csv" pattern
    # The glob regex * matches characters, so this finds e.g., "CO_1_iso.csv"
    all_files = list(Path(ROOT_DIR).rglob("*_1_iso.csv"))

    print(f"Found {len(all_files)} potential files.")

    for file_path in all_files:
        # Ignore readme files
        if "readme" in str(file_path).lower():
            continue

        # Ignore files that are ALREADY named "1_iso.csv" (we only want to process the source files)
        if file_path.name == "1_iso.csv":
            continue

        # --- Name Extraction Logic ---
        # Input: "CO_1_iso.csv" -> Split on "_" -> "CO"
        filename = file_path.name
        mol_name = filename.split('_')[0]

        # --- Check Flags ---
        if not should_process_molecule(mol_name):
            continue

        print(f"Processing: {mol_name} (Source: {filename})...")

        try:
            df = pd.read_csv(file_path)

            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)

            if mol_name not in MOLECULAR_DATA:
                print(f"  [WARNING] No metadata found for {mol_name}. Skipping.")
                continue

            mol_props = MOLECULAR_DATA[mol_name]

            # 1. Assign Basic Constants
            df['molecule_dipole'] = mol_props['dipole']
            df['self_dipole'] = mol_props['dipole']
            df['polar'] = mol_props['polar']
            df['B0a'] = mol_props['B0a']
            df['B0b'] = mol_props['B0b']
            df['B0c'] = mol_props['B0c']
            df['d'] = mol_props['d']

            if 'molecule_quadrupole' in mol_props:
                df['molecule_quadrupole_x'] = mol_props['molecule_quadrupole']

            # 2. Add Global Constants
            df = apply_constants_to_df(df, CONSTANTS)

            # 3. Calculate D-Air
            d_air_val = 364 * 0.8 + 346 * 0.2
            df['dair'] = d_air_val
            df['findair'] = (df['d'] + df['dair']) / 2

            # 4. Apply Symmetry
            df['symmetry'] = mol_props['symmetry']
            df = approximate_quantum_numbers(df, mol_props['symmetry'])

            # 5. Determine 'm'
            if mol_name == 'CH4':
                df['m'] = 10
            elif mol_name in ['C2N2', 'C4H2']:
                df['m'] = 8
            else:
                df['m'] = determine_multipole_m(df)

            # 6. Save Logic (New Filename)
            # Input: .../CO/CO_1_iso.csv
            # Output: .../CO/1_iso.csv
            output_path = file_path.parent / "1_iso.csv"

            df.to_csv(output_path, index=False)
            print(f"  -> Saved {len(df)} rows to {output_path}")

        except Exception as e:
            print(f"  [ERROR] Failed processing {mol_name}: {e}")


if __name__ == "__main__":
    main()