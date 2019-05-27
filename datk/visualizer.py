import numpy as np

def put_coords(subplot, data, labels, annotate=True):
    data = np.array(data)
    labels = np.array(labels)
    for i, label in enumerate(set(labels)):
        indices = np.where(labels == label)[0]
        group = data[indices, :]
        subplot.scatter(group[:, 0], group[:, 1], label=label, s=50, alpha=0.7)
        if annotate:
            subplot.annotate(label, xy=(group[0][0], group[0][1]), xytext=(-5, 5), textcoords='offset points')
    subplot.legend(loc='best')
    # subplot.set_xticks([])
    # subplot.set_yticks([])
