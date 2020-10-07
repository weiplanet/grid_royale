from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math
import inspect
import re
import abc
import random
import itertools
import collections.abc
import statistics
import concurrent.futures
import enum
import functools
import numbers
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable)
import dataclasses

import more_itertools
import keras.models
import tensorflow as tf
import numpy as np

from .strategizing import Strategy, QStrategy
from .base import Observation, Action, ActionObservation
from . import utils


class ModelBasedEpisodicLearningStrategy(Strategy):
    def __init__(self, curiosity: numbers.Real = 2, gamma: numbers.Real = 0.9) -> None:
        self.q_map = QMap()
        self.curiosity = curiosity
        self.gamma = gamma
        self.action_observation_chains_lists = collections.defaultdict(list)


    def decide_action_for_observation(self, observation: Observation,
                                       extra: Any = None) -> Action:
        action = max(
            observation.legal_actions,
            key=lambda action: self.q_map.get_ucb(
                observation, action, curiosity=self.curiosity
            )
        )
        return action

    def train(self, observation: Observation, action: Action,
              next_observation: Observation) -> None:

        action_observation_chains = self.action_observation_chains_lists[observation]
        try:
            action_observation_chain = action_observation_chains.pop()
        except IndexError:
            action_observation_chain = [ActionObservation(None, observation)]

        action_observation_chain.append(ActionObservation(action, next_observation))

        if next_observation.is_end:
            del self.action_observation_chains_lists[]
            self.q_map.add_sample(observation, action, q)
        else:
            self.action_observation_chains_lists[next_observation].append(action_observation_chain)





    # def get_observation_v(self, observation: Observation) -> numbers.Real:
        # raise NotImplementedError
        # return max(self.get_q_for_observation_action(observation, action) for action in
                   # observation.legal_actions)




def _zero_maker():
    return 0


class QMap(collections.abc.Mapping):
    def __init__(self) -> None:
        self._q_values = collections.defaultdict(_zero_maker)
        self._n_samples = collections.defaultdict(_zero_maker)
        self.n_total_samples = 0

    __len__ = lambda self: len(self._q_values)
    __iter__ = lambda self: iter(self._q_values)

    def __getitem__(self, observation_and_action: Iterable) -> numbers.Real:
        return self._q_values[self._to_key(*observation_and_action)]

    def _to_key(self, observation: Observation, action: Action) -> Tuple(bytes, Action):
        return (observation.to_neurons().tobytes(), action)


    def add_sample(self, observation: Observation, action: Action, q: numbers.Real) -> None:
        key = self._to_key(observation, action)
        self._q_values[key] = (
            self._q_values[key] *
            (self._n_samples[key] / (self._n_samples[key] + 1)) +
            q * (1 / (self._n_samples[key] + 1))
        )
        self._n_samples[key] += 1
        self.n_total_samples += 1

    def get_ucb(self, observation: Observation, action: Action,
                curiosity: numbers.Real) -> numbers.Real:
        key = self._to_key(observation, action)
        return self._q_values[key] + curiosity * math.sqrt(
            utils.cute_div(
                math.log(self.n_total_samples + 2),
                self._n_samples[key]
            )
        )


