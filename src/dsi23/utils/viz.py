import numpy as np
import matplotlib.pyplot as plt


def plot_curves(all_data, cutoff):
    nmdls = len(all_data)
    plt.figure()
    y_max = 0
    for i in range(nmdls):
        plt.plot(all_data[i].curve.t, all_data[i].curve.y, c='black')
        y_max = max(y_max, np.max(all_data[i].curve.y))
    plt.vlines(cutoff, 0, y_max, colors='red', linestyles='dashed')
    plt.show()
