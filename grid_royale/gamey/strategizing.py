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

from .base import Observation, Action, ActionObservation, SinglePlayerCulture
from . import utils




class Strategy(abc.ABC):

    State: Type[State]

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
                action_observation.observation.state.get_next_observation(action)
            )

    def __repr__(self) -> str:
        return f'{type(self).__name__}{self._extra_repr()}'

    def _extra_repr(self) -> str:
        return ('(?)' if inspect.signature(self.__init__).parameters else '()')

    def train(self, observation: Observation, action: Action,
              next_observation: Observation) -> None:
        pass # Put your training logic here, if your strategy requires training.


class SinglePlayerStrategy(Strategy):

    def get_score(self, n: int = 1_000, state_factory: Optional[Callable] = None,
                  gamma: Optional[numbers.Real] = None,
                  max_length: Optional[int] = None) -> int:

        single_player_culture = SinglePlayerCulture(self)
        single_player_culture.iterate_many_games(n=n, max_length=max_length,
                                                 state_factory=state_factory)
        make_state = (self.State.make_initial() if state_factory is None else state_factory)
        gamma_ = self.gamma if gamma is None else gamma
        return sum(
            sum(
                (gamma_ ** i) * action_observation.observation.reward
                for i, action_observation in
                enumerate(
                    more_itertools.islice_extended(
                        self.iterate_game(make_state())
                    )[:max_length]
                )
            )
            for _ in range(n)
        )





class RandomStrategy(Strategy):
    def decide_action_for_observation(self, observation: Observation, *,
                                       extra: Any = None) -> Action:
        return random.choice(observation.legal_actions)


class QStrategy(Strategy):
    @abc.abstractmethod
    def get_observation_v(self, observation: Observation) -> numbers.Real:
        raise NotImplementedError

    @abc.abstractmethod
    def get_qs_for_observations(self, observations: Sequence[Observation]) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        raise NotImplementedError

    def get_qs_for_observation(self, observation: Observation) -> Mapping[Action, numbers.Real]:
        return more_itertools.one(self.get_qs_for_observations((observation,)))




