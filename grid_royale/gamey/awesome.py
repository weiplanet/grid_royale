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
import numpy as np

from .base import StateActionReward, PlayerState, Action, ActionPlayerState
from .strategizing import Strategy, NiceStrategy
from . import utils

def _fit_external(model: keras.Model, *args, **kwargs) -> list:
    model.fit(*args, **kwargs)
    return model.get_weights()

class TrainingData:
    def __init__(self, awesome_strategy: AwesomeStrategy, max_size=10_000) -> None:
        self.awesome_strategy = awesome_strategy
        self.max_size = max_size
        self.counter = 0
        self._last_trained_batch = 0
        self.old_player_state_neuron_array = np.zeros(
            (max_size, awesome_strategy.player_state_type.n_neurons)
        )
        self.new_player_state_neuron_array = np.zeros(
            (max_size, awesome_strategy.player_state_type.n_neurons)
        )
        self.action_neuron_array = np.zeros(
            (max_size,awesome_strategy.player_state_type.action_type.n_neurons)
        )
        self.reward_array = np.zeros(max_size)
        self.are_not_end_array = np.zeros(max_size)

    def add(self, old_player_state: PlayerState, action: Action,
            new_player_state: PlayerState) -> None:
        self.old_player_state_neuron_array[self.counter_modulo] = old_player_state.to_neurons()
        self.action_neuron_array[self.counter_modulo] = action.to_neurons()
        self.new_player_state_neuron_array[self.counter_modulo] = new_player_state.to_neurons()
        self.reward_array[self.counter_modulo] = getattr(new_player_state,
                                                         self.awesome_strategy.reward_name)
        self.are_not_end_array[self.counter_modulo] = int(not new_player_state.is_end)
        self.counter += 1


    def is_training_time(self) -> bool:
        n_batches = self.counter // self.awesome_strategy.training_batch_size
        return n_batches > self._last_trained_batch


    def mark_trained(self) -> None:
        self._last_trained_batch = self.counter // self.awesome_strategy.training_batch_size
        assert not self.is_training_time()

    @property
    def counter_modulo(self) -> int:
        return self.counter % self.max_size

    @property
    def filled_max_size(self) -> bool:
        return self.counter >= self.max_size






class AwesomeStrategy(NiceStrategy):
    def __init__(self, player_state_type: Type[PlayerState], *, epsilon: numbers.Real = 0.3,
                 gamma: numbers.Real = 0.9, training_batch_size: int = 100,
                 loss: str = 'mse', optimizer: str = 'rmsprop', n_epochs: int = 50) -> None:
        NiceStrategy.__init__(self, player_state_type=player_state_type)
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_epochs = n_epochs
        self._fit_future: Optional[concurrent.futures.Future] = None

        self.model = keras.models.Sequential(
            layers=(
                keras.layers.Dense(
                    128, activation='relu',
                    input_dim=player_state_type.n_neurons
                ),
                keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(
                    128, activation='relu',
                ),
                keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(
                    128, activation='relu',
                ),
                keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(
                     player_state_type.action_type.n_neurons, # activation='relu'
                ),

            ),
            name='awesome_model'
        )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.training_batch_size = training_batch_size
        self.training_data = TrainingData(self)



    def train(self, executor: Optional[concurrent.futures.Executor] = None) -> None:
        n_actions = len(self.player_state_type.action_type)
        slicer = ((lambda x: x) if self.training_data.filled_max_size else
                  (lambda x: x[:self.training_data.counter_modulo]))
        old_player_state_neurons = slicer(self.training_data.old_player_state_neuron_array)
        new_player_state_neurons = slicer(self.training_data.new_player_state_neuron_array)
        action_neurons = slicer(self.training_data.action_neuron_array)
        are_not_ends = slicer(self.training_data.are_not_end_array)
        rewards = slicer(self.training_data.reward_array)
        n_data_points = old_player_state_neurons.shape[0]

        if self._fit_future is not None:
            weights = self._fit_future.result()
            self.model.set_weights(weights)
            self._fit_future = None

        old_fuck, new_fuck = np.split(
            self.model.predict(
                np.concatenate((old_player_state_neurons, new_player_state_neurons))
            ),
            2
        )

        # Assumes discrete actions:
        action_indices = np.dot(action_neurons, range(n_actions)).astype(np.int32)

        batch_index = np.arange(n_data_points, dtype=np.int32)
        old_fuck[batch_index, action_indices] = (
            rewards + self.gamma * np.max(new_fuck, axis=1) * are_not_ends
        )

        fit_arguments = {
            'x': old_player_state_neurons,
            'y': old_fuck,
            'epochs': max(1, int(self.n_epochs *
                                 (n_data_points / self.training_data.max_size))),
            'verbose': 0,
        }

        # This seems not to work fast:
        # if executor is not None:
            # self._fit_future = executor.submit(_fit_external, self.model, **fit_arguments)
        # else:
        self.model.fit(**fit_arguments)

        self.training_data.mark_trained()


    def get_qs_for_player_states(self, player_states: Optional[Sequence[PlayerState]] = None, *,
                                 player_states_neurons: Optional[np.ndarray] = None) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        if player_states is None:
            assert player_states_neurons is not None
            input_array = player_states_neurons
            check_action_legality = False
        else:
            assert player_states_neurons is None
            input_array = np.concatenate(
                [player_state.to_neurons()[np.newaxis, :] for player_state in player_states]
            )
            check_action_legality = True
        prediction_output = self.model.predict(input_array)
        actions = self.player_state_type.action_type
        if check_action_legality:
            return tuple(
                {action: q for action, q in dict(zip(actions, output_row)).items()
                 if (action in player_state.legal_actions)}
                for player_state, output_row in zip(player_states, prediction_output)
            )
        else:
            return tuple(
                {action: q for action, q in dict(zip(actions, output_row)).items()}
                for output_row in prediction_output
            )



    def decide_action_for_player_state(self, player_state: PlayerState, *,
                                       forced_epsilon: Optional[numbers.Real] = None,
                                       extra: Optional[np.ndarray] = None) -> Action:
        epsilon = self.epsilon if forced_epsilon is None else forced_epsilon
        if 0 < epsilon > random.random():
            return random.choice(player_state.legal_actions)
        else:
            q_map = self.get_qs_for_player_state(player_state) if extra is None else extra
            return max(q_map, key=q_map.__getitem__)


    def get_player_state_v(self, player_state: PlayerState,
                           epsilon: Optional[numbers.Real] = None) -> numbers.Real:
        if epsilon is None:
            epsilon = self.epsilon
        q_map = self.get_qs_for_player_state(player_state)
        return np.average(
            (
                max(q_map.values()),
                np.average(tuple(q_map.values()))
            ),
            weights=(1 - epsilon, epsilon)
        )


    def iterate_game(self, player_state: PlayerState) -> Iterator[ActionPlayerState]:
        iterator = utils.iterate_windowed_pairs(Strategy.iterate_game(self, player_state))
        yield ActionPlayerState(None, player_state)
        for old_action_player_state, new_action_player_state in iterator:
            self.training_data.add(old_action_player_state.player_state,
                                   *new_action_player_state)
            if self.training_data.is_training_time():
                self.train()
            yield new_action_player_state


