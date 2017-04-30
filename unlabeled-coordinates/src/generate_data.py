"""This is a mini script that generates data to experiment with.

Although I will be using unsupervised classification algorithm, the generated
data is labeled so that we could measure the classifier accuracy.
"""

from typing import Tuple
import random
import itertools

import click

import fs


class PointGenerator:
    def __call__(self) -> Tuple[float, float, int]:
        """
        Returns:
            (x, y, class): random point with label.
        """
        raise Exception('Not implemented')


class PointsSplitByYAxis:
    def __call__(self) -> Tuple[float, float, int]:
        cls = random.randint(0, 1)
        if cls == 0:
            x_bounds = (-20, -2)
        else:
            x_bounds = (2, 20)
        x = random.uniform(*x_bounds)
        y = random.uniform(0, 20)
        return (x, y, cls)


@click.command()
@click.option('--samples', default=1000,
              help='Number of generated data samples.')
@click.option('--output', default='vectors.json',
              help='File to write json data to.')
def main(samples: int, output: str) -> None:
    random_data = make_n_vectors(samples, PointsSplitByYAxis())
    fs.write_json_to(output, random_data, pretty=True)


def make_n_vectors(n: int, point_generator: PointGenerator) -> list:
    return list(itertools.islice(generate_data(point_generator), n))


def generate_data(point_generator) -> dict:
    while True:
        x, y, point_class = point_generator()
        yield {'x': x, 'y': y, 'class': point_class}


if __name__ == '__main__':
    main()
