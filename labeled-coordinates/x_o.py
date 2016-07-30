#
# Identify X and O points on a coordinate plane.
#
#    o        ^
#        o    |
#      o      |
#             |
# ------------+------------->
#             |    x
#             |       x
#             |  x


from sklearn.neighbors import KNeighborsClassifier

import coords


def main():
    clf = KNeighborsClassifier()
    clf.fit(*training_data(coords.make_n_random(100)))
    print(clf.predict([[1, -7], [-1, 7]]))


def training_data(coordinates):
    """Converts coordinates to classifier acceptable format."""
    return (
        [[x, y] for x, y, _ in coordinates],
        [label for _, _, label in coordinates]
    )


main()
