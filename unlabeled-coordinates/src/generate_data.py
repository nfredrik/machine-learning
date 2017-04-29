"""This is a mini script that generates data to experiment with.

Although I will be using unsupervised classification algorithm, the generated
data is labeled so that we could measure the classifier accuracy.
"""

from typing import Tuple
import random
import itertools

import click

import fs


Point = Tuple[float, float]


class LinearFunction:
    """a*x + b"""

    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b

    def __call__(self, x: int) -> float:
        return x * self.a + self.b


@click.command()
@click.option('--samples', default=1000,
              help='Number of generated data samples.')
@click.option('--output', default='vectors.json',
              help='File to write json data to.')
def main(samples: int, output: str) -> None:
    random_data = make_n_vectors(samples, LinearFunction(1, 0))
    fs.write_json_to(output, random_data, pretty=True)


def make_n_vectors(n: int, classification_fn: LinearFunction) -> list:
    return [item for item in
            itertools.islice(gen_linear_data(classification_fn), n)]


def gen_linear_data(classification_fn: LinearFunction) -> dict:
    while True:
        x, y = random_point()
        point_class = classify_point((x, y), classification_fn)
        yield {'x': x, 'y': y, 'class': point_class}


def classify_point(point: Point, classification_fn: LinearFunction) -> int:
    """Simple point classification using a linear funcion.

    Linear function divides points into two classes.

    Returns:
        Class of a point: either 1 or 2.
    """
    x = point[0]
    y = point[1]
    if y > classification_fn(x):
        return 1
    return 2


def random_point() -> Point:
    return (random.uniform(-20, 20), random.uniform(0, 20))


if __name__ == '__main__':
    main()
