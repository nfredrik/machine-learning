import random


def random_y_for(x):
    min_y = 0
    max_y = 10

    if x >= 0:
        min_y = -10
        max_y = 0

    return random.randint(min_y, max_y)


def label_for(x):
    return 'o' if x < 0 else 'x'


def make_random():
    x = random.randint(-10, 10)
    y = random_y_for(x)

    return (x, y, label_for(x))


def make_n_random(n):
    return [make_random() for _ in range(0, n)]


def all_labels():
    return ['x', 'o']
