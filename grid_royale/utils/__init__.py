import time as time_module
import threading
import copyreg
import contextlib
from typing import Iterable, Iterator, Hashable

import tensorflow.python._tf_stack

from . import make_keras_picklable


def iterate_deduplicated(iterable: Iterable[Hashable], seen: Iterable[Hashable] = ()) \
                                                                              -> Iterator[Hashable]:
    seen = set(seen)
    for item in iterable:
        if item in seen:
            continue
        else:
            yield item
            seen.add(item)


def pickle_lock(lock):
    return (threading.Lock, ())

copyreg.pickle(type(threading.Lock()), pickle_lock)


def pickle_r_lock(r_lock):
    return (threading.RLock, ())

copyreg.pickle(type(threading.RLock()), pickle_r_lock)


def pickle_stack_summary(stack_summary):
    return (tensorflow.python._tf_stack.StackSummary, ())


copyreg.pickle(tensorflow.python._tf_stack.StackSummary, pickle_stack_summary)
