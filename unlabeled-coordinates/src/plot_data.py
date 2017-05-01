import matplotlib
import matplotlib.pyplot as plt

import click

import fs


@click.command()
@click.option('--in', 'data_file', default='vectors.json',
              help='Input data file path.')
@click.option('--out', 'output_image', default='data.svg',
              help='Output image file.')
def main(data_file: str, output_image: str) -> None:
    coords = fs.read_json_from(data_file)
    generate_image_from(coords, output_image)


def generate_image_from(coords: list, img_path: str) -> None:
    ax = plt.subplot(111)
    for c in coords:
        ax.plot(c['x'], c['y'], 'o')
    plt.savefig(img_path)
    plt.gcf().clear()


if __name__ == '__main__':
    main()
