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
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import dataclasses

import more_itertools
import keras.models
import tensorflow as tf
import numpy as np

from .utils import ImmutableDict
from . import exceptions




class _NiceDataclass(collections.abc.Sequence):
    __len__ = lambda self: len(dataclasses.fields(self))
    __iter__ = lambda self: map(
        self.__dict__.__getitem__,
        (field.name for field in dataclasses.fields(self))
    )
    __getitem__ = lambda self, i: tuple(self)[i]

@dataclasses.dataclass(order=True, frozen=True)
class ActionObservation(_NiceDataclass):
    action: Optional[Action]
    observation: Observation


class _ActionType(abc.ABCMeta):# collections.abc.Sequence):
    __iter__ = lambda cls: iter(cls.all_actions)
    __len__ = lambda cls: len(cls.all_actions)
    def __getitem__(cls, i: int):
        if i >= len(cls):
            raise IndexError
        for j, item in enumerate(cls):
            if j == i:
                return cls
        raise RuntimeError

    @property
    def n_neurons(cls) -> int:
        try:
            return cls._n_neurons
        except AttributeError:
            cls._n_neurons = len(cls)
            return cls._n_neurons



_action_regex_head = re.compile(r'[A-Za-z0-9.]')
_action_regex_tail = re.compile(r'[A-Za-z0-9_.\-/>]*')
_action_regex = re.compile(f'^{_action_regex_head.pattern}'
                           f'{_action_regex_tail.pattern}$')

@functools.total_ordering
class Action(metaclass=_ActionType):
    all_actions: Sequence[Action]
    n_neurons: int

    def __lt__(self, other):
        return self.all_actions.index(self) < self.all_actions.index(other)

    def slugify(self) -> str:
        raw = str(self)
        first_letter = raw[0]
        prefix = '' if _action_regex_head.fullmatch(first_letter) else '0'
        characters = ((c if _action_regex_tail.fullmatch(c) else '-') for c in raw)
        result = f'{prefix}{"".join(characters)}'
        assert _action_regex.fullmatch(result)
        return result

    def to_neurons(self) -> np.ndarray:
        # Implementation for simple discrete actions. Can override.
        try:
            return self._to_neurons
        except AttributeError:
            self._to_neurons = np.array([int(self == action) for action in type(self)],
                                        dtype=np.float64)
            return self._to_neurons

    @classmethod
    def from_neurons(cls, neurons: Iterable) -> Action:
        # Implementation for simple discrete actions. Can override.
        return cls[tuple(neurons).index(1)]


class Observation(abc.ABC):
    state: State
    legal_actions: Tuple[Action, ...]
    is_end: bool
    reward: numbers.Real
    n_neurons: int

    @abc.abstractmethod
    def to_neurons(self) -> np.ndarray:
        raise NotImplementedError

PlayerId = TypeVar('PlayerId', bound=Hashable)

class State(abc.ABC):
    Observation: Type[Observation]
    Action: Type[Action]
    is_end: bool
    player_id_to_observation: ImmutableDict[PlayerId, Observation]

    @abc.abstractmethod
    def get_next_state_from_actions(self, player_id_to_action: Mapping[PlayerId, Action]) -> State:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def make_initial() -> State:
        raise NotImplementedError


class _SinglePlayerStateType(abc.ABCMeta):
    @property
    def Observation(cls) -> _SinglePlayerStateType:
        return cls


class SinglePlayerState(State, Observation, metaclass=_SinglePlayerStateType):

    player_id_to_observation = property(lambda self: ImmutableDict({None: self}))


    @abc.abstractmethod
    def get_next_state_from_action(self, action: Action) -> SinglePlayerState:
        raise NotImplementedError

    def get_next_state_from_actions(self, player_id_to_action: Mapping[PlayerId, Action]) \
                                                                               -> SinglePlayerState:
        return self.get_next_state_from_action(more_itertools.one(player_id_to_action.values()))


class Culture:
    def __init__(self, state_type: Type[State],
                 player_id_to_strategy: Mapping[PlayerId, strategizing.Strategy]) -> None:
        self.State = state_type
        self.player_id_to_strategy = player_id_to_strategy


    def iterate_many_games(self, *, n: int = 10, max_length: int = 100,
                           state_factory: Optional[Callable] = None) -> Iterator[State]:
        state_factory = ((lambda: self.State.make_initial()) if state_factory is None
                         else state_factory)
        for i in range(n):
            state: State = state_factory()
            yield from self.iterate_game(state, max_length)


    def iterate_game(self, state: State, max_length: Optional[int] = None) -> Iterator[State]:
        yield state
        iterator = range(1, max_length) if max_length is not None else itertools.count(1)
        for i in iterator:
            if state.is_end:
                return
            state = self.get_next_state(state)
            yield state


    def get_next_state(self, state: State) -> State:
        if state.is_end:
            raise exceptions.GameOver
        player_id_to_action = {
            player_id: self.player_id_to_strategy[player_id
                                                        ].decide_action_for_observation(observation)
            for player_id, observation in state.player_id_to_observation.items()
            if not observation.is_end
        }
        next_state = state.get_next_state_from_actions(player_id_to_action)
        for player_id, action in player_id_to_action.items():
            strategy = self.player_id_to_strategy[player_id]
            observation = state.player_id_to_observation[player_id]
            strategy.train(observation, action, next_state.player_id_to_observation[player_id])
        return next_state


class SinglePlayerCulture(Culture):

    def __init__(self, state_type: Type[SinglePlayerState], *,
                 strategy: strategizing.Strategy) -> None:
        self.strategy = strategy
        Culture.__init__(self, state_type=state_type,
                         player_id_to_strategy=ImmutableDict({None: strategy}))




from . import strategizing

