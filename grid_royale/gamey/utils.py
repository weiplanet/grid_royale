from __future__ import generator_stop

import numpy as np
from typing import Optional, Union, Tuple, Iterable, Iterator
import numbers

import more_itertools
from immutabledict import immutabledict as ImmutableDict


class LastDetectingIterator(more_itertools.peekable):
    def on_last_item(self):
        try:
            self.peek()
        except StopIteration:
            return True
        else:
            return False


def cute_div(x: numbers.Number, y: numbers.Number) -> numbers.Number:
    try:
        return x / y
    except ZeroDivisionError:
        if x == 0:
            raise
        else:
            return x * float('inf')

def clamp(number, /, minimum, maximum):
    if number < minimum:
        return minimum
    elif number > maximum:
        return maximum
    else:
        return number


def iterate_windowed_pairs(iterable: Iterable) -> Iterator:
    iterator = iter(iterable)
    try:
        old = next(iterator)
    except StopIteration:
        return
    for new in iterator:
        yield (old, new)
        old = new
