from __future__ import annotations

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
                    Sequence, Callable, Mapping)
import dataclasses

import more_itertools
import keras.models
import tensorflow as tf
import numpy as np

from .base import Observation, Action, StateActionReward, ActionObservation
from . import utils




class Strategy(abc.ABC):

    gamma: numbers.Real = 1
    reward_name: str = 'reward'
    Game: Type[Game]

    def get_score(self, n: int = 1_000, observation_factory: Optional[Callable] = None,
                  forced_gamma: Optional[numbers.Real] = None,
                  max_game_length: Optional[int] = None) -> int:
        make_observation = (self.Game.Observation.make_initial if observation_factory is None
                      else observation_factory)
        gamma = self.gamma if not forced_gamma else forced_gamma
        return sum(
            sum(
                (gamma ** i) * getattr(action_observation.observation, self.reward_name)
                for i, action_observation in
                enumerate(
                    more_itertools.islice_extended(
                        self.iterate_game(make_observation())
                    )[:max_game_length]
                )
            )
            for _ in range(n)
        )

    @abc.abstractmethod
    def decide_action_for_observation(self, observation: Observation,
                                       extra: Any = None) -> Action:
        raise NotImplementedError

    def iterate_game(self, observation: Observation) -> Iterator[ActionObservation]:
        action_observation = ActionObservation(None, observation)
        while True:
            yield action_observation
            if action_observation.observation.is_end:
                return
            action_observation = ActionObservation(
                (action := self.decide_action_for_observation(action_observation.observation)),
                action_observation.observation.get_next_observation(action)
            )

    def __repr__(self) -> str:
        return f'{type(self).__name__}{self._extra_repr()}'

    def _extra_repr(self) -> str:
        return ('(?)' if inspect.signature(self.__init__).parameters else '()')


class RandomStrategy(Strategy):
    def decide_action_for_observation(self, observation: Observation, *,
                                       extra: Any = None) -> Action:
        return random.choice(observation.legal_actions)


class NiceStrategy(Strategy):
    @abc.abstractmethod
    def get_observation_v(self, observation: Observation) -> numbers.Real:
        raise NotImplementedError

    @abc.abstractmethod
    def get_qs_for_observations(self, observations: Sequence[Observation]) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        raise NotImplementedError

    def get_qs_for_observation(self, observation: Observation) -> Mapping[Action, numbers.Real]:
        return more_itertools.one(self.get_qs_for_observations((observation,)))




