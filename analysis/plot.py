import os
import sys

import pickle
import matplotlib.pyplot as plt
import numpy as np


def cdf_and_pdf(data,
                bins=20,
                range=None,
                density=None,
                label_scale=1,
                rotation=30,
                percentile=False,
                percent=0.02,
                ):

    percentage = len(data[(data >= range[0]) & (data <= range[1])]) / len(data)

    hist, bin_edges = np.histogram(data, bins=bins, density=density, range=range)

    cdf = np.cumsum(hist / sum(hist) * percentage)
#     if isinstance(bins, (int, float)) and bins > 20:
#         plt.xticks(bin_edges[2::2])
#     else:
    plt.xticks(bin_edges[1:])
    plt.tick_params(axis='x', labelsize=10 * label_scale)  # , labelrotation=rotation)
    plt.plot(bin_edges[1:], cdf, '-*', alpha=0.5)# color='#ED7D31')

    width = -(bin_edges[1] - bin_edges[0])
    plt.bar(bin_edges[1:], hist / sum(hist),
            width=width,
            align='edge',
            alpha=0.5,)
            # color='#5B9BD5')

    x_shift = width * 0.5
    y_shift = 0.02
    for i, (a, b) in enumerate(zip(bin_edges[1:], hist / sum(hist))):
        if a < range[0] or a > range[1]:
            continue
        plt.text(a + x_shift, b + y_shift, '%.3f' % b,
                 ha='center', va='bottom', fontsize=10 * label_scale, rotation=rotation)
    for i, (a, b) in enumerate(zip(bin_edges[1:], cdf)):
        if a < range[0] or a > range[1]:
            continue
        plt.text(a + x_shift, b + y_shift, '%.3f' % b,# color='#ED7D31',
                 ha='center', va='bottom', fontsize=10 * label_scale, rotation=rotation)

    if percentile:
        if not percentile in ['min', 'max', 'both']:
            raise ValueError('percentile must be in "min", "max" or "both"')
        min_value = np.percentile(data, percent * 100)
        max_value = np.percentile(data, (1 - percent) * 100)
        if percentile == 'min':
            plt.axvline(x=min_value, ls='-.',)# color='#66CCFF')
            plt.text(min_value, 0.8, f'{percent*100}%\npercentile:\n{min_value:.2f}',
                     ha='center', va='bottom', fontsize=12 * label_scale)
        if percentile == 'max':
            plt.axvline(x=max_value, ls='-.',)# color='#66CCFF')
            plt.text(max_value, 0.8, f'{(1-percent)*100}%\npercentile:\n{max_value:.2f}',
                     ha='center', va='bottom', fontsize=12 * label_scale)
        if percentile == 'both':
            plt.axvline(x=min_value, ls='-.',) #color='#66CCFF')
            plt.text(min_value, 0.8, f'{percent*100}%\npercentile:\n{min_value:.2f}',
                     ha='center', va='bottom', fontsize=12 * label_scale)
            plt.axvline(x=max_value, ls='-.',) #color='#66CCFF')
            plt.text(max_value, 0.8, f'{(1-percent)*100}%\npercentile:\n{max_value:.2f}',
                     ha='center', va='bottom', fontsize=12 * label_scale)

    plt.xlim(range)

if __name__ == '__main__':
    _dir = sys.argv[1]
    files = sys.argv[2].split(',')

    plt.figure(figsize=(6, 4), dpi=150)
    plt.title(sys.argv[3], fontsize='large', fontweight='bold')

    legends = []
    for i, p in enumerate(files):
        data = pickle.load(open(os.path.join(_dir, p), 'rb'))
        cdf_and_pdf(data, range=(0, int(sys.argv[4])), bins=20)

        legends.append(p.split('_')[-1][:-4])

    plt.legend(legends)

    plt.show()
