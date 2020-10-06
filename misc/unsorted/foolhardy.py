import pathlib
import json
import itertools
import numbers
from typing import Sequence

from matplotlib import pyplot as plt
import more_itertools
import scipy.stats
import numpy as np


games_folder: pathlib.Path = (pathlib.Path(__file__).parent / 'pal' / 'games')
paths = sorted(games_folder.rglob('*.json'))

foo = []
for path in paths:
    with path.open('r') as file:
        states = json.load(file)
    for state in states:
        foo.append(sum(player_data[2] < -1 for player_data in state['players']))

def smooth(sequence: Sequence[numbers.Real], radius: int = 10) -> Sequence[numbers.Real]:
    raw_convolution = scipy.stats.norm.pdf(np.arange(-radius, radius + 1), 0, radius / 3)
    convolution = raw_convolution / sum(raw_convolution)
    return [sum(window * convolution) for window in
            more_itertools.windowed(sequence, n=(2 * radius + 1))]

plt.plot(smooth(foo, radius=300))

plt.show()