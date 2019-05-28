import numpy as np
from matplotlib.lines import Line2D


def put_coords(subplot, data, labels, annotate=True, legend=True):
    data = np.array(data)
    labels = np.array(labels)
    for i, label in enumerate(set(labels)):
        indices = np.where(labels == label)[0]
        group = data[indices, :]
        subplot.scatter(group[:, 0], group[:, 1], label=label, s=50, alpha=0.7)
        if annotate:
            subplot.annotate(label, xy=(group[0][0], group[0][1]), xytext=(-5, 5), textcoords='offset points')
    if legend:
        subplot.legend(loc='best')
    # subplot.set_xticks([])
    # subplot.set_yticks([])


def legend(subplot, labels, colors):
    lines = []
    for color in colors:
        lines.append(Line2D([0], [0], linewidth=7.0, linestyle='-', color=color))
    subplot.legend(lines, labels, loc='best', borderpad=0.7)


def put_nsigma(subplot, x, mu, sigma, n=3, color='r', alpha=0.1, label=''):
    mu = np.array(mu)
    sigma = np.array(sigma)

    subplot.plot(x, mu, color, label=label)
    for i in range(1, n + 1):
        subplot.fill_between(x, mu - sigma * i, mu + sigma * i, color=color, alpha=alpha)
