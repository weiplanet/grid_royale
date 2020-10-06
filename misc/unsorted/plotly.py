import pathlib
import json
import itertools
import numbers
from typing import Sequence

from matplotlib import pyplot as plt
import more_itertools
import scipy.stats
import numpy as np


def smooth(sequence: Sequence[numbers.Real], /, radius: int = 10) -> Sequence[numbers.Real]:
    raw_convolution = scipy.stats.norm.pdf(np.arange(-radius, radius + 1), 0, radius / 3)
    convolution = raw_convolution / sum(raw_convolution)
    return [sum(window * convolution) for window in
            more_itertools.windowed(sequence, n=(2 * radius + 1))]



# text = pathlib.Path('nice-munch.csv').read_text()
text = pathlib.Path('aggro-266528.csv').read_text()

sequences = tuple(zip(*(map(float, line.split(',')) for line in text.splitlines())))

smooth_sequences = [smooth(sequence, radius=5) for sequence in sequences]

for i, sequence in enumerate(smooth_sequences, start=1):
    plt.plot(sequence, label=f'{i}-multiplier')

plt.legend()

plt.show()