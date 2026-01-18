import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    '/Users/elizabeth/Desktop/line_broadening.nosync/'
    'Scratch/choose_features_Compare_runs/run_comparison_summary.csv'
)

# start plotting from the first remove_QN entry
start_idx = df[df['run'].str.contains('remove_QN', na=False)].index[0]
df2 = df.iloc[start_idx:].copy()

# baseline reference value
baseline = df[df['run'].str.contains('compression_0.5', na=False)]['rmse_mean'].iloc[0]

# mapping long names to short labels
mapping = {
    "remove_QN_J_Jpp": 'J\', J"',
    "remove_QN_Kas": 'Ka\', Ka"',
    "remove_QN_Kcs": 'Kc\', Kc"',
    "remove_QN_M": 'M (index)',
    "remove_b_a_b": 'B$_{0}$b (active)',
    "remove_b_p_b": 'B$_{0}$b (perturber)',
    "remove_d_act_per": 'd (combined diameter)',
    "remove_ds": 'ds (active & perturber)',
    "remove_is_self": "'is self' flag",
    "remove_jeanna": 'γ$_{B}$',
    "remove_m": 'm (combined pole)',
    "remove_ms": 'm (active & perturber)',
    "remove_quad_axy": 'XX and YY quadrupole (active)',
    "remove_quad_pxy": 'XX and YY quadrupole (perturber)',
    "remove_quad_use_RMSE": 'add in RMSE of quadrupoles',
    "remove_weight": 'mass',
    "total_feature_selection": 'keep 12 best features',
    "z_better_features_?": 'drop 10 worst features',
    "z_remove_M_Kas_ds": 'M + Ka + ds',
    "z_remove_M_Kas_ds_Jeanna": 'M Ka ds γ_B',
    "z_remove_M_Kas_ds_Quad+RMS": 'M Ka ds RMSE-Q1',
    "z_remove_M_Kas_ds_RMS": 'M Ka ds RMSE-Q2'
}

def shorten(name: str) -> str:
    suffix = name.split('2025-11-14_')[-1] if '2025-11-14_' in name else name
    print(suffix)
    return mapping.get(suffix, suffix)


df2['label'] = df2['run'].apply(shorten)

# use all runs from remove_QN onwards
# cut after "10 worst"
cut_idx = df2[df2['label'] == 'M + Ka + ds'].index.max()
df3 = df2.loc[:cut_idx].copy()
df3 = df3[df3['label'] != "add in RMSE of quadrupoles"]

plt.figure()


# baseline line
plt.axvline(baseline, linestyle='--', color='r',
            label='baseline - all features')
# main points
plt.scatter(df3['rmse_mean'], df3['label'],
            marker='x', color='b')#, label='runs')

# circle the 6 smallest RMSE points
smallest6 = df3.nsmallest(6, 'rmse_mean')
smallest6 = smallest6[smallest6['label'] != 'γ$_{B}$']
smallest6 = smallest6[smallest6['label'] != 'drop 10 worst features']

plt.scatter(smallest6['rmse_mean'], smallest6['label'],
            facecolors='none', edgecolors='g', s=200,
            label='Changes kept')

plt.legend()
plt.xlabel('mean RMSE (cm$^{-1}$ atm$^{-1}$)')
plt.ylabel('Feature removed')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('feature comparison.png')
