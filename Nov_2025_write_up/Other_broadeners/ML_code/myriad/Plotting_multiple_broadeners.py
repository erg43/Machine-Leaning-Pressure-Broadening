import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open('output_Trial_multiple_broadeners_pipe_LLM_100X_data_9_21.pkl', 'rb') as f:
    pipe = pickle.load(f)

with open('output_Trial_multiple_broadeners_data_list_LLM_100X_data_9_21.pkl', 'rb') as f:
    data_list = pickle.load(f)

cmap = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
        '#6a3d9a', '#ffff99', '#b15928']

for item in data_list:
    key = item[0]
    x_plot_glob = item[1]
    y_test_plot_glob = item[2]
    y_pred_glob = item[3]
    score = item[4]
    mse_score = item[5]
    active_molecule_weight = item[6]
    active_molecule_key = item[7]
    active_molecule_errors = 1 / np.sqrt(item[8])
    print(key)
    print(score)
    print(mse_score)

    for weight in active_molecule_weight.unique():
        active_molecule = active_molecule_key[active_molecule_weight == weight]
        x = x_plot_glob[active_molecule_weight == weight]
        y_test = y_test_plot_glob[active_molecule_weight == weight]
        y_pred = y_pred_glob[active_molecule_weight == weight]
        data_error = np.mean(active_molecule_errors[active_molecule_weight == weight])

        plt.figure(figsize=((15, 6)), dpi=500)
        fig = plt.figure(1)
        plt.rc('font', size=18)

        plt.plot(x, y_test, 'x', color=cmap[1], label='Literature $\gamma$ data')
        plt.plot(x, y_pred, '.', color=cmap[5], label="$\gamma$ predicted by Voting Model")

        #plt.title(
        #    f'Comparison of \gamma from machine learning results against HITRAN data values, shown for {key, active_molecule}')
        plt.title(active_molecule.iloc[0])
        plt.xlabel('m')
        plt.ylabel(f'\gamma /cm$^{{{-1}}}$atm$^{{{-1}}}$')
        plt.ylim(0.00, max([max(y_pred), max(y_test)]) + 0.01)
        plt.errorbar(max(x + 3), np.mean(y_test), yerr=data_error * np.mean(y_test), color=cmap[1])
        print(active_molecule.iloc[0])
        print(data_error * np.mean(y_test))

        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=2)
        plt.savefig(f'plots/9_21_{active_molecule.iloc[0]}.png')
        plt.close(fig)