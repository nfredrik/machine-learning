from itertools import tee

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

import coords


def main():
    plot_coords(coords.make_n_random(100))


def coords_for_label(coords, label):
    coords_x, coords_y = tee(
        filter(lambda x_y_label: x_y_label[2] == label, coords))
    return ([x for x, _, _ in coords_x], [y for _, y, _ in coords_y])


def plot_coords(coordinates):
    ax = plt.subplot(111)

    for label in coords.all_labels():
        x, y = coords_for_label(coordinates, label)
        ax.plot(x, y, label)

    plt.show()


main()
