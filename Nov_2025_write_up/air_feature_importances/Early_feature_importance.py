import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
from joblib import dump, load  # <--- Add this import

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---- CONFIGURATION ----
# Adjust this path to where your folders (CO, NO, etc.) are located
BASE_DIR = Path.home() / "Desktop" / "line_broadening.nosync" / "line_broadening" / "hitran_data"
PLOT_DIR = Path.home() / "Desktop" / "line_broadening.nosync" / "line_broadening" / "Nov_2025_write_up" / "air feature importances"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
save_dir = PLOT_DIR / "saved_models"

# Features used for training (X)
FEATURES = [
    'nu', 'sw', 'a', 'v', 'J', 'vpp', 'Jpp',
    'molecule_weight', 'air_weight', 'molecule_dipole',
    'B0', 'coord', 'polar', 'open_shell', 'wexe',
    'mass_ratio', 'S', 'Spp', 'Omega', 'Omegapp',
    'Lambda', 'Lambdapp', 'm', 'findair'
]

TARGET = 'gamma_air'
WEIGHT_COL = 'gamma_air-err'


# ---- DATA LOADING ----
def load_and_preprocess_data(base_path):
    """
    Loads CSVs, initializes missing physics columns, and returns a dict of DataFrames.
    """
    # Map molecule names to their specific file paths based on your structure
    # Assuming structure: base_path/CO/1_iso.csv
    files = {
        "CO": base_path / "CO/1_iso.csv",
        "NO": base_path / "NO/1_iso.csv",
        "OH": base_path / "OH/1_iso.csv",
        "HF": base_path / "HF/1_iso.csv",
        "HCl": base_path / "HCl/1_iso.csv",
        "HBr": base_path / "HBr/1_iso.csv",
        "HI": base_path / "HI/1_iso.csv",
        "ClO": base_path / "ClO/1_iso.csv",
        "CS": base_path / "CS/1_iso.csv",
        "SO": base_path / "SO/1_iso.csv"
    }
    molecules = {}
    #'molecule_weight', 'air_weight', 'molecule_dipole', 'B0', 'coord', 'polar', 'open_shell', 'wexe', 'mass_ratio', 'm', 'findair']
    #"['molecule_weight', 'B0', 'coord', 'open_shell', 'wexe', 'mass_ratio']
    for name, filepath in files.items():
        if not filepath.exists():
            print(f"Warning: File not found for {name} at {filepath}. Skipping.")
            continue

        df = pd.read_csv(filepath)

        # Clean unnamed columns
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        # Initialize missing columns with 0 if they don't exist
        for col in ['Lambda', 'Lambdapp', 'Omega', 'Omegapp', 'S', 'Spp']:
            if col not in df.columns:
                df[col] = 0

        if name == 'CO':
            df['molecule_weight'] =28.01
            df['B0'] =1.9225
            df['coord'] =1.128
            df['open_shell'] =0
            df['wexe'] =13.29
            df['mass_ratio'] =0.7502

        molecules[name] = df
        print(f"Loaded {name}: {df.shape}")

    return molecules


# ---- TRAINING & EVALUATION ----
def train_and_evaluate(molecules):
    """
    Performs Leave-One-Molecule-Out Cross-Validation.
    Returns aggregated feature importances.
    """

    # Setup save directory
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    # Store importances for aggregation
    mdi_importances = []
    perm_importances = []
    scores = []

    molecule_names = list(molecules.keys())

    print("\nStarting Leave-One-Molecule-Out Training...")
    print("-" * 60)

    for test_mol in molecule_names:
        print(f"Testing {test_mol}")
        # 1. Split Data
        train_mols = [m for m in molecule_names if m != test_mol]

        # Concatenate training data
        df_train = pd.concat([molecules[m] for m in train_mols], ignore_index=True)
        # Get test data
        df_test = molecules[test_mol].copy()

        # 2. Prepare X, y, and weights
        # Ensure all features exist
        missing_feats = [f for f in FEATURES if f not in df_train.columns]
        if missing_feats:
            raise ValueError(f"Missing features in data: {missing_feats}")

        X_train = df_train[FEATURES]
        y_train = df_train[TARGET]
        w_train = df_train[WEIGHT_COL]  # Use error code as sample weight

        X_test = df_test[FEATURES]
        y_test = df_test[TARGET]
        w_test = df_test[WEIGHT_COL]

        # 3. Define Pipeline
        # Using the logic from your first snippet
        pipe = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(random_state=42)
        )

        # 4. Fit
        # Pass sample weights to the regressor step
        pipe.fit(X_train, y_train, gradientboostingregressor__sample_weight=w_train)

        # 4. Save the Model
        # We name it 'model_test_CO.joblib' implying this is the model used
        # when CO was the TEST set (trained on everything else).
        if save_dir:
            model_filename = save_path / f"model_test_{test_mol}.joblib"
            dump(pipe, model_filename)
            # print(f"Saved: {model_filename.name}")

        # 5. Score
        score = pipe.score(X_test, y_test, sample_weight=w_test)
        scores.append(score)
        print(f"Test on {test_mol:<4} | Train size: {len(X_train):<6} | R2 Score: {score:.4f}")

        # 6. Extract MDI Feature Importance (Model based)
        # Steps[1] is the regressor
        model = pipe.steps[1][1]
        mdi_importances.append(model.feature_importances_)

        # 7. Extract Permutation Importance (Test data based)
        # This tells us which features matter for generalizing to the NEW molecule
        result = permutation_importance(
            pipe, X_test, y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        perm_importances.append(result.importances_mean)

    print("-" * 60)
    print(f"Average R2 Score: {np.mean(scores):.4f}")

    return np.array(mdi_importances), np.array(perm_importances)


# ... (Imports remain the same)

# ---- UPDATED PLOTTING FUNCTION ----
def plot_aggregated_importance(importances, title, filename, top_k=30):
    """
    Plots the Mean importance +/- Std Dev.
    Matches the style of the 'feature_importances_mean_std' plot:
    - Standard matplotlib style (no outlines)
    - Top K features only
    - Inverted Y-axis (Top feature at top)
    """
    mean_imp = importances.mean(axis=0)
    std_imp = importances.std(axis=0)

    # 1. Sort High to Low and slice Top K
    # argsort gives low-to-high, so we reverse it with [::-1]
    order = np.argsort(mean_imp)[-top_k:][::-1]

    # 2. Setup Plot
    plt.figure(figsize=(10, 8))
    ypos = np.arange(len(order))

    # 3. Create Horizontal Bar Plot
    # Note: No 'color' or 'edgecolor' specified to match standard style
    plt.barh(ypos, mean_imp[order], xerr=std_imp[order], align="center", capsize=3)

    # 4. Formatting
    plt.yticks(ypos, np.array(FEATURES)[order])
    plt.xlabel("Mean Importance")
    plt.title(f"{title}\n(Mean ± Std across folds)")

    # Invert axis so index 0 (Highest Importance) is at the top
    plt.gca().invert_yaxis()

    plt.tight_layout()

    # 5. Save
    save_path = PLOT_DIR / filename
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")
    plt.show()
    plt.close()  # Close to free memory

# ---- MAIN EXECUTION ----
if __name__ == "__main__":

    # 1. Load Data
    data_dict = load_and_preprocess_data(BASE_DIR)

    if not data_dict:
        print("No data loaded. Check your paths.")
    else:
        # 2. Train Models and get Importances
        mdi_vals, perm_vals = train_and_evaluate(data_dict)

        # 3. Plot MDI Importance (What the model relied on during training)
        plot_aggregated_importance(
            mdi_vals,
            "Feature Importance",
            "feature_importance_MDI.png"
        )

        # 4. Plot Permutation Importance (What mattered for prediction on new molecules)
        plot_aggregated_importance(
            perm_vals,
            "Permutation Importance (on Test Set)",
            "feature_importance_Permutation.png"
        )