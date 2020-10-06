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

from .base import PlayerState, Action, StateActionReward, ActionPlayerState
from . import utils




class Strategy(abc.ABC):

    gamma: numbers.Real = 1
    reward_name: str = 'reward'

    def __init__(self, player_state_type: Type[PlayerState]) -> None:
        self.player_state_type = player_state_type

    def get_score(self, n: int = 1_000, player_state_factory: Optional[Callable] = None,
                  forced_gamma: Optional[numbers.Real] = None,
                  max_game_length: Optional[int] = None) -> int:
        make_player_state = (self.player_state_type.make_initial if player_state_factory is None
                      else player_state_factory)
        gamma = self.gamma if not forced_gamma else forced_gamma
        return sum(
            sum(
                (gamma ** i) * getattr(action_player_state.player_state, self.reward_name)
                for i, action_player_state in
                enumerate(
                    more_itertools.islice_extended(
                        self.iterate_game(make_player_state())
                    )[:max_game_length]
                )
            )
            for _ in range(n)
        )

    @abc.abstractmethod
    def decide_action_for_player_state(self, player_state: PlayerState,
                                       extra: Any = None) -> Action:
        raise NotImplementedError

    def iterate_game(self, player_state: PlayerState) -> Iterator[ActionPlayerState]:
        action_player_state = ActionPlayerState(None, player_state)
        while True:
            yield action_player_state
            if action_player_state.player_state.is_end:
                return
            action_player_state = ActionPlayerState(
                (action := self.decide_action_for_player_state(action_player_state.player_state)),
                action_player_state.player_state.get_next_player_state(action)
            )

    def __repr__(self) -> str:
        return f'{type(self).__name__}{self._extra_repr()}'

    def _extra_repr(self) -> str:
        return ('(?)' if inspect.signature(self.__init__).parameters else '()')


class RandomStrategy(Strategy):
    def decide_action_for_player_state(self, player_state: PlayerState, *,
                                       extra: Any = None) -> Action:
        return random.choice(player_state.legal_actions)


class NiceStrategy(Strategy):
    @abc.abstractmethod
    def get_player_state_v(self, player_state: PlayerState) -> numbers.Real:
        raise NotImplementedError

    @abc.abstractmethod
    def get_qs_for_player_states(self, player_states: Sequence[PlayerState]) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        raise NotImplementedError

    def get_qs_for_player_state(self, player_state: PlayerState) -> Mapping[Action, numbers.Real]:
        return more_itertools.one(self.get_qs_for_player_states((player_state,)))




