#!python
import pathlib
import statistics
import json
import numbers
from typing import Sequence, Union, Any

import click
import more_itertools
import scipy.stats
from matplotlib import pyplot as plt
import numpy as np


def smooth(sequence: Sequence[numbers.Real], /, radius: int = 10) -> Sequence[numbers.Real]:
    raw_convolution = scipy.stats.norm.pdf(np.arange(-radius, radius + 1), 0, radius / 3)
    convolution = raw_convolution / sum(raw_convolution)
    return [sum(window * convolution) for window in
            more_itertools.windowed(sequence, n=(2 * radius + 1))]


@click.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('-r', '--radius', type=int, default=10)
def flucker(folder, radius):
    folder = pathlib.Path(folder)
    json_paths = tuple(sorted(folder.glob('*.json')))
    average_rewards = []
    for json_path in json_paths:
        with json_path.open() as file:
            states = json.load(file)
        for state in states:
            average_rewards.append(
                statistics.mean(reward for _, _, reward, _, _ in state['players'])
            )

    plt.plot(smooth(average_rewards, radius=radius))
    plt.show()


if __name__ == '__main__':
    flucker()
