import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import math
import warnings

# --- User Configuration ---
BASE = (
        Path.home()
        / "Desktop"
        / "line_broadening.nosync"
        / "Scratch"
        / "other_broadeners_2025-11-14_new_baseline_try_comp_0"
)

def plot_feature_dependence(model, data_for_stats, feat_cols, save_path: Path):
    """
    Plots predicted Gamma vs Feature Range for all features,
    holding other features at their mean.

    Args:
        model: The trained scikit-learn model/pipeline.
        data_for_stats (pd.DataFrame): DataFrame used to calculate feature
                                       means and ranges (e.g., data_train or data_test).
        feat_cols (list[str]): List of feature names in the correct order.
        save_path (Path): Path to save the figure to.
    """

    print(f"Generating feature dependence plots -> {save_path} ...")

    # 1. Setup the grid for subplots
    n_features = len(feat_cols)
    n_cols = 4  # Number of columns in the plot grid
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        constrained_layout=True
    )
    axes = axes.flatten()  # Flatten to easily iterate

    # 2. Calculate means for all features (the "holding" values)
    try:
        means = data_for_stats[feat_cols].mean()
    except KeyError:
        print("Error: Not all features in feat_cols found in data_for_stats.")
        print(f"Model features: {feat_cols}")
        print(f"Data columns: {data_for_stats.columns.tolist()}")
        plt.close(fig)
        return

    # 3. Iterate through each feature
    for i, feature in enumerate(feat_cols):
        ax = axes[i]

        # A. Determine the range for this specific feature
        x_min = data_for_stats[feature].min()
        x_max = data_for_stats[feature].max()

        # Plot on log scale if 'B0' is in the name and all data is positive
        epsilon = 1e-9
        is_log_scale = 'B0' in feature and (x_min > epsilon)

        # Generate points across the feature's range
        if data_for_stats[feature].nunique() < 20:
            # For discrete features, use their unique values
            x_vals = np.sort(data_for_stats[feature].unique())
        elif is_log_scale:
            x_vals = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        else:
            # For continuous features, sample 100 points
            x_vals = np.linspace(x_min, x_max, 100)

        # B. Construct the synthetic input matrix
        # Start by tiling the mean vector
        X_synthetic = np.tile(means.values, (len(x_vals), 1))

        # Create a DataFrame to ensure column order is respected
        X_synthetic_df = pd.DataFrame(X_synthetic, columns=feat_cols)

        # Replace the column for the current feature with its range
        X_synthetic_df[feature] = x_vals

        # C. Predict
        # The model expects a numpy array matching feat_cols order
        # We use .to_numpy() on the DataFrame to guarantee this
        try:
            y_pred_log = model.predict(X_synthetic_df[feat_cols].to_numpy())
            # Model was trained on log(gamma), so apply exp()
            y_pred = np.exp(y_pred_log)
        except Exception as e:
            print(f"Error predicting for feature {feature}: {e}")
            ax.set_title(f"{feature}\n(Prediction Error)", fontsize=10, color='red')
            continue

        # D. Plotting
        ax.plot(x_vals, y_pred, color='royalblue', linewidth=2)
        ax.set_title(feature, fontsize=10, fontweight='bold')
        ax.set_ylabel(r"$\gamma$ (predicted)", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

        if is_log_scale:
            ax.set_xscale('log')

        # Add a rug plot to show data distribution
        # Use a small, random sample to avoid overplotting
        sample_size = min(500, len(data_for_stats))
        sample_data = data_for_stats[feature].sample(sample_size, random_state=42)

        # Calculate y-position for rug plot (e.g., 5% from bottom)
        ymin, ymax = ax.get_ylim()
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            ymin, ymax = np.nanmin(y_pred), np.nanmax(y_pred)
            if not np.isfinite(ymin) or not np.isfinite(ymax):
                ymin, ymax = 0, 1

        rug_y_pos = ymin + (ymax - ymin) * 0.05
        ax.plot(sample_data, np.full_like(sample_data, rug_y_pos),
                '|', color='k', alpha=0.2, markersize=10)

    # 4. Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        r"Model Sensitivity: Output $\gamma$ vs Feature (others fixed at mean)",
        fontsize=16
    )

    # Save the figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved feature dependence plots to: {save_path.resolve()}")
    plt.close(fig)


def main():
    """
    Load all .joblib models in BASE, and generate feature-dependence plots
    for each model using the same data_for_stats.
    """

    data_filename = "synthetic_feature_dataset.csv"
    # --------------------------

    # 1. Load Data for Stats (shared across all models)
    data_path = data_filename

    print(f"Loading data for stats from {data_path}...")
    try:
        data_for_stats = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Iterate over all .joblib models in BASE
    model_paths = sorted(BASE.glob("*.joblib"))
    if not model_paths:
        print(f"No .joblib models found in {BASE}")
        return

    for model_path in model_paths:
        print(f"\nProcessing model: {model_path.name}")

        # Load model
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model {model_path.name}: {e}")
            continue

        # Get feature columns
        try:
            feat_cols = model.feature_names_in_
        except AttributeError:
            print(
                "Warning: 'feature_names_in_' not found on model. "
                "Inferring features from data."
            )
            drop = {
                "gamma", "gamma_uncertainty", "fractional_error", "profile",
                "paper", "weight", "pair", "M", "Ka_aprox", "Kapp_aprox",
                "active_d", "perturber_d", "T", "Unnamed: 0"
            }
            feat_cols = [
                c for c in data_for_stats.columns
                if c not in drop
                and np.issubdtype(data_for_stats[c].dtype, np.number)
            ]

        if not feat_cols:
            print(f"Error: Could not determine feature columns for {model_path.name}.")
            continue

        # 3. Generate plots for this model
        out_png = Path("feature_dependence_plots") / BASE.name / f"{model_path.stem[:20]}_feature_dependence.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_feature_dependence(model, data_for_stats, feat_cols, out_png)


if __name__ == "__main__":
    main()
