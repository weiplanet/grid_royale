from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import enum
import functools
import pickle
import threading
import json
import pathlib
import string as string_module
import statistics
import math
import concurrent.futures
import itertools
import random
import logging
import operator
import numbers
import io
import collections
from typing import (Optional, Tuple, Union, Container, Hashable, Iterator,
                    Iterable, Any, Dict, FrozenSet, Callable, Type)
import dataclasses
import datetime as datetime_module

import numpy as np
import scipy.special
import more_itertools

from . import gamey
from . import utils
from .gamey.utils import ImmutableDict
from .vectoring import Vector, Step, Position, Translation, Vicinity

SHOT_REWARD = -10
COLLISION_REWARD = -5
FOOD_REWARD = 10
NOTHING_REWARD = -1
VISION_RANGE = 4
VISION_SIZE = VISION_RANGE * 2 + 1

N_CORE_STRATEGIES = 1

LETTERS = string_module.ascii_uppercase

current_folder = pathlib.Path(__file__).parent

_action_neuron_cache = {}

@dataclasses.dataclass(order=True, frozen=True)
class Action(gamey.Action):
    move: Optional[Step]
    shoot: Optional[Step]

    def __post_init__(self):
        if tuple(self).count(None) != 1:
            raise ValueError

    __iter__ = lambda self: iter((self.move, self.shoot))


    def _to_neurons(self) -> np.ndarray:
        result = np.zeros(8, dtype=np.float64)
        result[Action.all_actions.index(self)] = 1
        return result

    def to_neurons(self) -> np.ndarray:
        return _action_neuron_cache[self]

    @property
    def name(self) -> str:
        if self.move is not None:
            return self.move.name
        else:
            return f'shoot_{self.shoot.name}'



(Action.up, Action.right, Action.down, Action.left) = \
    Action.all_move_actions = (Action(Step.up, None), Action(Step.right, None),
                                   Action(Step.down, None), Action(Step.left, None))

(Action.shoot_up, Action.shoot_right, Action.shoot_down, Action.shoot_left) = \
    Action.all_shoot_actions = (Action(None, Step.up), Action(None, Step.right),
                                   Action(None, Step.down), Action(None, Step.left))

Action.all_actions = Action.all_move_actions + Action.all_shoot_actions

for action in Action:
    _action_neuron_cache[action] = action._to_neurons()


@dataclasses.dataclass(order=True, frozen=True)
class Bullet:
    position: Position
    direction: Step

    def get_next_bullet(self):
        return Bullet(position=(self.position + self.direction), direction=self.direction)




class _BaseGrid:
    board_size: int
    def __contains__(self, position: Position) -> bool:
        return 0 <= min(position) <= max(position) <= self.board_size - 1

    @staticmethod
    def iterate_random_positions(board_size: int) -> Iterator[Position]:
        while True:
            yield Position(
                random.randint(0, board_size - 1),
                random.randint(0, board_size - 1),
            )



class State(_BaseGrid, gamey.State):

    def __init__(self, grid_royale_culture: GridRoyaleCulture, *, board_size: int,
                 letter_to_observation: ImmutableDict[str, Observation],
                 food_positions: FrozenSet[Position],
                 bullets: ImmutableDict[Position, FrozenSet[Bullet]] = ImmutableDict(),
                 be_training: bool = True) -> None:
        self.grid_royale_culture = grid_royale_culture
        self.letter_to_observation = letter_to_observation
        self.bullets = bullets
        assert all(bullets.values()) # No empty sets there.
        self.all_bullets = frozenset(itertools.chain.from_iterable(bullets.values()))
        self.food_positions = food_positions
        self.board_size = board_size
        self.living_player_positions = ImmutableDict(
            {observation.position: observation for observation in letter_to_observation.values()
             if not observation.is_end}
        )
        self.is_end = not self.living_player_positions
        self.be_training = be_training

    def _reduce(self) -> tuple:
        return (
            type(self),
            frozenset(
                (letter, observation.position, observation.score, observation.reward,
                 observation.last_action) for letter, observation in
                self.letter_to_observation.items()
            ),
            self.bullets, self.food_positions, self.board_size, self.is_end
        )

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, State) and self._reduce() == other._reduce()

    def __hash__(self) -> int:
        return hash(self._reduce())


    @staticmethod
    def make_initial(grid_royale: GridRoyaleCulture, board_size: int = 24,
                     starting_score: int = 10 ** 6, concurrent_food_tiles: int = 40,
                     be_training: bool = True) -> State:

        n_players = len(strategies)
        random_positions_firehose = utils.iterate_deduplicated(
                                     State.iterate_random_positions(board_size=board_size))
        random_positions = tuple(
            more_itertools.islice_extended(
                random_positions_firehose)[:(n_players + concurrent_food_tiles)]
        )

        player_positions = random_positions[:n_players]
        food_positions = frozenset(random_positions[n_players:])
        assert len(food_positions) == concurrent_food_tiles

        letter_to_observation = {}
        for player_position, letter in zip(player_positions, LETTERS):
            letter_to_observation[letter] = Observation(state=None, position=player_position,
                                                        score=starting_score, letter=letter,
                                                        last_action=None),

        state = State(
            grid_royale=grid_royale_culture,
            board_size=board_size,
            letter_to_observation=ImmutableDict(letter_to_observation),
            food_positions=food_positions,
            be_training=be_training,
        )

        for observation in letter_to_observation.values():
            observation.state = state

        return state


    def get_next_state_from_actions(self, player_id_to_action: Mapping[Position, Action]) \
                                                                                  -> State:
        new_player_position_to_olds = collections.defaultdict(set)
        for letter, action in player_id_to_action.items():
            action: Action
            old_observation = self.letter_to_observation[letter]
            assert action in old_observation.legal_actions
            old_player_position = old_observation.position
            if action.move is not None:
                new_player_position_to_olds[old_player_position +
                                                               action.move].add(old_player_position)
            else:
                new_player_position_to_olds[old_player_position].add(old_player_position)

        ############################################################################################
        ### Figuring out which players collided into each other: ###################################
        #                                                                                          #
        # There are three types of collisions:
        # 1. Two or more players that try to move into the same position.
        # 2. Two players that are trying to move into each other's positions.
        # 3. Any players that are trying to move into the old position of a player that had one
        #    of the two collisions above, and is therefore still occupying that position.

        collided_player_positions = set()
        while True:
            for new_player_position, old_player_positions in new_player_position_to_olds.items():
                if len(old_player_positions) >= 2:
                    # Yeehaw, we have a collision! This is either type 1 or type 3. Let's punish
                    # everyone!
                    collided_player_positions |= old_player_positions
                    del new_player_position_to_olds[new_player_position]
                    for old_player_position in old_player_positions:
                        new_player_position_to_olds[old_player_position].add(old_player_position)

                    # We modified the dict while iterating, let's restart the loop:
                    break

                elif (len(old_player_positions) == 1 and
                    ((old_player_position := more_itertools.one(old_player_positions)) !=
                      new_player_position) and new_player_position_to_olds.get(
                                               old_player_position, None) == {new_player_position}):
                    collided_player_positions |= {old_player_position, new_player_position}
                    new_player_position_to_olds[new_player_position] = {new_player_position}
                    new_player_position_to_olds[old_player_position] = {old_player_position}

                    # We modified the dict while iterating, let's restart the loop:
                    break

            else:
                # We already found all collisions, if any.
                break
        #                                                                                          #
        ### Finished figuring out which players collided into each other. ##########################
        ############################################################################################

        new_player_position_to_old = {
            new_player_position: more_itertools.one(old_player_positions) for
            new_player_position, old_player_positions in new_player_position_to_olds.items()
        }
        del new_player_position_to_olds # Prevent confusion

        ############################################################################################
        ### Figuring out bullets: ##################################################################
        #                                                                                          #

        # Todo: This section needs a lot of tests!

        wip_bullets = collections.defaultdict(set)

        # Continuing trajectory for existing bullets:
        for bullet in self.all_bullets:
            new_bullet = bullet.get_next_bullet()
            wip_bullets[new_bullet.position].add(new_bullet)

        # Processing new bullets that were just fired:
        for player_position, action in player_id_to_action.items():
            if action.shoot is not None:
                new_bullet = Bullet(player_position + action.shoot, action.shoot)
                wip_bullets[new_bullet.position].add(new_bullet)

        # Clearing bullets out of board:
        for position in [position for position, bullets in wip_bullets.items()
                         if (position not in self)]:
            del wip_bullets[position]

        # Figuring out which players were shot, removing these bullets:
        new_player_positions_that_were_shot = set()
        for new_player_position, old_player_position in new_player_position_to_old.items():
            if wip_bullets.get(new_player_position, None):
                # Common shooting case.
                del wip_bullets[new_player_position]
                new_player_positions_that_were_shot.add(new_player_position)
            elif ((move := player_id_to_action[old_player_position].move) and
                  (oncoming_bullets := set(bullet for bullet in wip_bullets.get(
                                           old_player_position, ()) if
                                           bullet.direction == (- move)))):
                # Less-common shooting case: The player walked towards an oncoming bullet, switching
                # places with it.
                (oncoming_bullet,) = oncoming_bullets
                wip_bullets[old_player_position].remove(oncoming_bullet)
                new_player_positions_that_were_shot.add(new_player_position)


        bullets = ImmutableDict({
            position: frozenset(bullets) for position, bullets in wip_bullets.items() if bullets
        })

        #                                                                                          #
        ### Finished figuring out bullets. #########################################################
        ############################################################################################

        ############################################################################################
        ### Figuring out food: #####################################################################
        #                                                                                          #
        new_player_positions_that_ate_food = set()
        wip_food_positions = set(self.food_positions)
        random_positions_firehose = utils.iterate_deduplicated(
            State.iterate_random_positions(board_size=self.board_size),
            seen=(set(new_player_position_to_old) |
                   self.living_player_positions | self.food_positions)
        )
        for new_player_position in new_player_position_to_old:
            if new_player_position in self.food_positions:
                wip_food_positions.remove(new_player_position)
                wip_food_positions.add(next(random_positions_firehose))
                new_player_positions_that_ate_food.add(new_player_position)

        assert len(wip_food_positions) == len(self.food_positions)

        #                                                                                          #
        ### Finished figuring out food. ############################################################
        ############################################################################################

        letter_to_observation = {}

        for new_player_position, old_player_position in new_player_position_to_old.items():
            letter = self.living_player_positions[old_player_position].letter
            old_observation: Observation = self.letter_to_observation[letter]

            reward = (
                SHOT_REWARD if new_player_position in new_player_positions_that_were_shot else
                COLLISION_REWARD if new_player_position in collided_player_positions else
                FOOD_REWARD if new_player_position in new_player_positions_that_ate_food else
                NOTHING_REWARD
            )

            letter_to_observation[letter] = Observation(
                state=None,
                position=new_player_position,
                score=old_observation.score + reward,
                reward=reward,
                letter=letter,
                last_action=player_id_to_action[letter]
            ),


        state = State(
            grid_royale=self.grid_royale_culture, board_size=self.board_size,
            letter_to_observation=ImmutableDict(letter_to_observation),
            food_positions=frozenset(wip_food_positions),
            be_training=self.be_training, bullets=bullets
        )

        for observation in letter_to_observation.values():
            observation.state = state

        return state



    def ascii(self):
        string_io = io.StringIO()
        for position in Position.iterate_all(self):
            if position.x == 0 and position.y != 0:
                string_io.write('|\n')
            if position in self.living_player_positions:
                observation = self.living_player_positions[position]
                letter = (observation.letter.lower() if observation.reward < NOTHING_REWARD
                          else observation.letter)
                string_io.write(letter)
            elif (bullets := self.bullets.get(position, None)):
                if len(bullets) >= 2:
                    string_io.write('Ӿ')
                else:
                    string_io.write(more_itertools.one(bullets).direction.ascii)


            elif position in self.food_positions: # But no bullets
                string_io.write('·')
            else:
                string_io.write(' ')

        string_io.write('|\n')
        string_io.write('‾' * self.board_size)
        string_io.write('\n')

        return string_io.getvalue()

    @property
    @functools.lru_cache()
    def serialized(self) -> dict:
        return {
            'players': [
                (
                    list(observation.position),
                    observation.score,
                    observation.reward,
                    observation.letter,
                    last_action.name if (last_action := observation.last_action) is not None
                    else None,
                 ) for letter, observation in self.letter_to_observation.items()
            ],
            'food_positions': list(map(list, self.food_positions)),
            'bullets': [
                [list(bullet.position), list(bullet.direction)] for bullet in self.all_bullets
            ],
        }

    def p(self) -> None:
        print(self.ascii)


    def write_to_folder(self, folder: pathlib.Path, *, chunk: int = 10,
                        max_length: Optional[int] = None, overwrite: bool = False):
        ### Preparing folder: #################################################
        #                                                                     #
        if not folder.exists():
            folder.mkdir(parents=True)
        assert folder.is_dir()
        if overwrite:
            for path in folder.iterdir():
                path.unlink()
        assert not any(folder.iterdir())
        #                                                                     #
        ### Finished preparing folder. ########################################

        paths = ((folder / f'{i:06d}.json') for i in range(10 ** 6))
        state_iterator = more_itertools.islice_extended(
                                                      self.iterate_states())[:max_length]
        for path in paths:
            state_chunk = []
            for state in more_itertools.islice_extended(state_iterator)[:chunk]:
                yield state
                state_chunk.append(state)
            if not state_chunk:
                return
            output = [state.serialized for state in state_chunk]
            with path.open('wb') as file:
                json.dump(output, file)

    def write_to_pal(self, *, chunk: int = 10, max_length: Optional[int] = None):
        games_folder: pathlib.Path = (current_folder.parent / 'pal' / 'games').resolve().absolute()
        now = datetime_module.datetime.now()
        game_folder_name = now.isoformat(sep='-', timespec='microseconds'
                                                               ).replace(':', '-').replace('.', '-')
        game_folder = games_folder / game_folder_name
        game_folder.mkdir(parents=True)
        print(f'Writing to {game_folder.name} ...')
        for state in self.write_to_folder(game_folder, chunk=chunk, max_length=max_length,
                                          overwrite=True):
            yield state
        print(f'Finished writing to {game_folder.name} .')


    @property
    def average_reward(self):
        return statistics.mean(observation.reward for observation in
                               self.letter_to_observation.values())






class Observation(_BaseGrid, gamey.Observation):

    def __init__(self, state: Optional[State], position: Position, *,
                 letter: str, score: int, last_action: Optional[Action],
                 reward: int = 0) -> None:
        assert len(letter) == 1
        self.state = state
        self.position = position
        self.letter = letter.upper()
        self.score = score
        self.reward = reward
        self.last_action = last_action
        self.is_end = (score <= 0)


    @property
    def legal_actions(self) -> Tuple[Action, ...]:
        if self.is_end:
            return ()
        actions = list(Action.all_shoot_actions)
        if self.position.y >= 1:
            actions.append(Action.up)
        if self.position.x <= self.board_size - 2:
            actions.append(Action.right)
        if self.position.y <= self.board_size - 2:
            actions.append(Action.down)
        if self.position.x >= 1:
            actions.append(Action.left)
        return tuple(actions)

    @property
    def legal_move_actions(self) -> Tuple[Action, ...]:
        return tuple(action for action in self.legal_actions if (action.move is not None))


    @property
    def board_size(self) -> int:
        return self.state.board_size

    @property
    def grid_royale_culture(self) -> GridRoyaleCulture:
        return self.state.grid_royale_culture

    n_neurons = (
        # + 1 # Score
        + 8 # Last action
        + 8 * ( # The following for each vicinity
            + 1 # Distance to closest player of each strategy
            + N_CORE_STRATEGIES # Distance to closest player of each strategy
            + 1 # Distance to closest food
            + 4 # Distance to closest bullet in each direction
            + 1 # Distance to closest wall
        )
        + VISION_SIZE ** 2 * ( # Simple vision
            + 1 # Wall
            + 1 # Food
            + 4 # Bullets in each direction
            + N_CORE_STRATEGIES # Players of each strategy
        )
    )

    @property
    def simple_vision(self):
        array = np.zeros((VISION_SIZE, VISION_SIZE, 1 + 1 + 4 + N_CORE_STRATEGIES), dtype=int)
        relative_player_position = Position(VISION_SIZE // 2, VISION_SIZE // 2)
        translation = relative_player_position - self.position

        for relative_position in Position.iterate_all(VISION_SIZE):
            absolute_position: Position = relative_position - translation
            if absolute_position not in self:
                array[tuple(relative_position) + (0,)] = 1
            elif absolute_position in self.state.food_positions:
                array[tuple(relative_position) + (1,)] = 1
            elif (bullets := self.state.bullets.get(absolute_position, None)):
                for bullet in bullets:
                    array[tuple(relative_position) + (2 + bullet.direction.index,)] = 1
            elif (observation := self.state.living_player_positions.get(absolute_position, None)):
                array[tuple(relative_position) +
                      (6 + self.grid_royale_culture.core_strategies.index(player.strategy),)] = 1

        return array

    @functools.lru_cache()
    def to_neurons(self) -> np.ndarray:
        last_action_neurons = (np.zeros(len(Action)) if self.last_action is None
                               else self.last_action.to_neurons())
        return np.concatenate((
            np.array(
                tuple(
                    itertools.chain.from_iterable(
                        self.processed_distances_to_food_players_bullets(vicinity) for
                        vicinity in Vicinity.all_vicinities
                    )
                )
            ),
            np.array(
                tuple(
                    self.processed_distance_to_wall(vicinity) for vicinity in
                    Vicinity.all_vicinities
                )
            ),
            last_action_neurons,
            self.simple_vision.flatten(),
            # (scipy.special.expit(self.score / 10),),
        ))

    _distance_base = 1.2

    @functools.lru_cache()
    def processed_distances_to_food_players_bullets(self, vicinity: Vicinity) -> numbers.Real:
        field_of_view = self.position.field_of_view(vicinity, self.board_size)

        for distance_to_food, positions in enumerate(field_of_view, start=1):
            if positions & self.state.food_positions:
                break
        else:
            distance_to_food = float('inf')


        distances_to_other_players = []

        for i, positions in enumerate(field_of_view, start=1):
            if positions & self.state.living_player_positions:
                distances_to_other_players.append(i)
                break
        else:
            distances_to_other_players.append(float('inf'))


        for strategy in self.grid_royale.core_strategies:
            for i, positions in enumerate(field_of_view, start=1):
                player_infos = (
                    self.state.player_infos.get(position, None) for position in
                    positions & self.state.living_player_positions
                )
                if strategy in (player_info.strategy for player_info in player_infos):
                    distances_to_other_players.append(i)
                    break
            else:
                distances_to_other_players.append(float('inf'))

        assert len(distances_to_other_players) == N_CORE_STRATEGIES + 1

        distances_to_bullets = []
        for direction in Step.all_steps:
            for i, positions in enumerate(field_of_view, start=1):
                bullets = itertools.chain.from_iterable(
                    self.state.bullets.get(position, ()) for position in positions
                )
                if any(bullet.direction == direction for bullet in bullets):
                    distances_to_bullets.append(i)
                    break
            else:
                distances_to_bullets.append(float('inf'))


        return tuple(self._distance_base ** (-distance)
                     for distance in ([distance_to_food] + distances_to_other_players +
                                      distances_to_bullets))


    @functools.lru_cache()
    def processed_distance_to_wall(self, vicinity: Vicinity) -> numbers.Real:
        position = self.position
        for i in itertools.count():
            if position not in self:
                distance_to_wall = i
                break
            position += vicinity
        return 1.2 ** (-distance_to_wall)

    @property
    def cute_score(self) -> int:
        return self.score - min((self.position @ food_position
                                for food_position in self.state.food_positions),
                                default=(-100))

    def p(self) -> None:
        print(self.ascii)

logging.getLogger('tensorflow').addFilter(
    lambda record: 'Tracing is expensive and the excessive' not in record.msg
)

class GridRoyaleCulture(gamey.Culture):

    def __init__(self, n_players: int = 20):
        self.core_strategies = tuple(Strategy(self) for _ in range(N_CORE_STRATEGIES))
        self.strategies = tuple(more_itertools.islice_extended(
                                                 itertools.cycle(self.core_strategies))[:n_players])
        self.executor = concurrent.futures.ProcessPoolExecutor(5)
        gamey.Culture.__init__(state_type=State,
                               player_id_to_strategy=dict(zip(LETTERS, strategies)))


    def grind(self, *, n: int = 10, max_length: int = 100) -> Iterator[State]:
        yield from super().grind(
            self.core_strategies, n=n, max_length=max_length,
            state_factory=self.make_initial
        )

    def make_initial(self) -> State:
        return State.make_initial(self, self.strategies)


class _GridRoyaleStrategy(gamey.Strategy):
    State = State


class SimpleStrategy(_GridRoyaleStrategy):

    def __init__(self, epsilon: int = 0.2):
        self.epsilon = epsilon


    def decide_action_for_observation(self, observation: Observation,
                                       extra: Any = None) -> Action:
        if random.random() <= self.epsilon or not observation.state.food_positions:
            return random.choice(observation.legal_actions)
        else:
            closest_food_position = min(
                (food_position for food_position in observation.state.food_positions),
                key=lambda food_position: observation.position @ food_position
            )
            desired_translation = closest_food_position - observation.position
            dimension = random.choice(
                tuple(dimension for dimension, delta in enumerate(desired_translation) if delta)
            )
            return (Action(np.sign(desired_translation.x), 0) if dimension == 0
                    else Action(0, np.sign(desired_translation.y)))



class Strategy(_GridRoyaleStrategy):

    def __init__(self, grid_royale: GridRoyale, **kwargs) -> None:
        self.grid_royale = grid_royale
        gamey.ModelFreeLearningStrategy.__init__(self, training_batch_size=10, **kwargs)

    def get_aggro(self, n_states: int = 100) -> numbers.Real:
        iterator = itertools.cycle(self.grid_royale.core_strategies)
        return self._analyze(
            lambda observation: observation.reward < NOTHING_REWARD,
            n_states=n_states, competing_strategy_factory=lambda: next(iterator)
        )

    def get_munch(self, n_states: int = 100) -> numbers.Real:
        return self._analyze(
            lambda observation: observation.reward == FOOD_REWARD,
            n_states=n_states
        )


    def _analyze(self, evaluate_state: Callable, n_states: int = 100,
                 n_other_players: int = 10,
                 competing_strategy_factory: Callable[[], gamey.Strategy]
                                   = lambda: gamey.RandomStrategy(Observation)) -> numbers.Real:
        with utils.time():
            strategies = [self] + [competing_strategy_factory()] * n_other_players
            n_observations = 0
            score = 0
            initial_states = [
                State.make_initial(self.grid_royale, strategies,
                                            starting_score=10 ** 6, be_training=False)
                for _ in range(3)
            ]
            iterator = more_itertools.interleave_longest(
                *(initial_state.iterate_states() for
                  initial_state in initial_states)
            )
            for state in iterator:
                state: State
                (observation,) = [
                    observation for player_info in state.player_infos.values()
                    if (observation := player_info.observation).letter == LETTERS[0]
                ]
                n_observations += 1
                score += evaluate_state(observation)
                if n_observations >= n_states:
                    return score / n_observations
            else:
                raise NotImplementedError

    @functools.lru_cache()
    def get_neurons_of_sample_states_and_best_actions(self) -> Tuple[np.ndarray,
                                                                     Tuple[Action, ...]]:

        def make_state(player_position: Position,
                       food_positions: Iterable[Position]) -> State:
            observation = Observation(None, player_position, letter=LETTERS[0],
                                                score=10 ** 6, reward=0, last_action=None)
            player_infos = ImmutableDict({
                player_position: PlayerInfo(id=player_position, observation=observation,
                                                strategy=self,)
            })
            state = State(
                self.grid_royale, board_size=24, player_infos=player_infos,
                food_positions=frozenset(food_positions), be_training=False
            )
            observation.state = state
            return state

        def make_states(player_position: Position,
                        food_positions: Iterable[Position]) -> State:
            return tuple(
                make_state(rotated_player_position, rotated_food_positions) for
                rotated_player_position, *rotated_food_positions in zip(
                    *map(lambda position: position.iterate_rotations_in_board(board_size=24),
                         itertools.chain((player_position,), food_positions))
                )
            )

        player_positions = [Position(x, y) for x, y in itertools.product((5, 11, 16), repeat=2)]
        states = []
        for player_position in player_positions:
            distances_lists = tuple(
                itertools.chain.from_iterable(
                    itertools.combinations(range(1, 4 + 1), i) for i in range(1, 3 + 1)
                )
            )
            for distances, step in itertools.product(distances_lists, Step.all_steps):
                states.extend(
                    make_states(player_position,
                                [player_position + distance * step for distance in distances])
                )

        observations = [more_itertools.one(state.player_infos.values()).observation
                         for state in states]

        def _get_cute_score_for_action(observation: Observation,
                                       legal_move_action: Action) -> int:
            next_state = observation.state. \
                   get_next_state_from_actions({observation.position: legal_move_action})
            return more_itertools.one(
                            next_state.player_infos.values()).observation.cute_score

        return tuple((
            np.concatenate([observation.to_neurons()[np.newaxis, :] for
                            observation in observations]),
            tuple(
                max(
                    observation.legal_move_actions,
                    key=lambda legal_action: _get_cute_score_for_action(observation, legal_action)
                )
                for observation in observations
            )
        ))

    def measure(self):
        neurons_of_sample_states, best_actions = \
                                                self.get_neurons_of_sample_states_and_best_actions()
        q_maps = self.get_qs_for_observations(observations_neurons=neurons_of_sample_states)
        decided_actions = tuple(max(q_map, key=q_map.__getitem__) for q_map in q_maps)
        return statistics.mean(decided_action == best_action for decided_action, best_action in
                               zip(decided_actions, best_actions))







state = None
strategies = None
grid_royale_culture = None

def run():
    global grid_royale_culture, state, strategies
    grid_royale_culture = GridRoyale()

    # for _ in grid_royale.train(n=2, max_length=100):
        # pass
    state = grid_royale_culture.make_initial()
    for i in range(2_000):
        # states = []
        for state in state.write_to_pal(chunk=50):
            # states.append(state)
            # print(f'{state.average_reward:.02f}')
            print(grid_royale_culture.core_strategies[0].measure())








