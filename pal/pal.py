import json
import math
import sys
import re
import time
import itertools
import numbers
import collections.abc
import urllib.parse
from typing import Union, Optional, Iterator, Iterable, TypeVar, Callable, Tuple, TypeVar
import random

from browser import document, html, ajax, timer, window


CELL_SIZE = window.CELL_SIZE
HALF_CELL_SIZE = window.HALF_CELL_SIZE
BOARD_WIDTH = window.BOARD_WIDTH
BOARD_HEIGHT = window.BOARD_HEIGHT

def clamp(number, /, minimum, maximum):
    assert minimum <= maximum
    if number < minimum:
        return minimum
    elif number > maximum:
        return maximum
    else:
        return number



class Favorites:
    timepoint_text = property(lambda self: document.select('#timepoint-text')[0])
    timepoint_slider = property(lambda self: document.select('#timepoint-slider')[0])
    speed_slider = property(lambda self: document.select('#speed-slider')[0])
    play_pause_button = property(lambda self: document.select('button#play-pause')[0])
    update_button = property(lambda self: document.select('button#update-button')[0])
    play_pause_button_img = property(lambda self: document.select('button#play-pause img')[0])
    game_selector = property(lambda self: document.select('#game-selector')[0])
    grid = property(lambda self: document.select('#grid')[0])
    bullets = property(lambda self: document.select('.bullet'))

    def _get_indexed(self, category, index, content):
        id = f'{category}-{index}'
        results = document.select(f'#{id}')
        if results:
            result, = results
            return result
        else:
            self.grid <= html.DIV(id=id, Class=category)
            results = document.select(f'#{id}')
            result, = results
            result.textContent = content
            return result

    def get_player(self, letter):
        return self._get_indexed('player', letter, letter)

    def get_food(self, position):
        x, y = position
        food = self._get_indexed('food', f'{x}-{y}', '·')
        food.classList.add(f'grid-column-{x + 1}')
        food.classList.add(f'grid-row-{y + 1}')
        return food


favorites = Favorites()


def iterate_x_y_cell_contents(state):
    for i, cell_content in enumerate(itertools.chain.from_iterable(state)):
        yield (*divmod(i, 24), cell_content)


def set_new_classes(item, *class_names):
    item.classList.value = ' '.join(class_names)

_directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
_directions_ascii = '↑→↓←'
_directions_names = ('up', 'right', 'down', 'left')

def _get_position_from_div(div: html.DIV) -> Tuple[int]:
    class_string = div.classList.value
    return (
        int(re.findall('grid-column-([0-9]+)', class_string)[0]) - 1,
        int(re.findall('grid-row-([0-9]+)', class_string)[0]) - 1,
    )


def update_button_handler(event=None) -> None:
    update_game_names()
    timeline.state_fetcher.update()



State = TypeVar('State', bound=list)

class Timeline(collections.abc.MutableSequence):

    def __init__(self) -> None:
        self._states = []
        self._funky_states = []
        self._needle = 0
        self._anchor_time = time.time()
        self._anchor_needle = self.needle
        self._anchor_speed = 3
        self._anchor_target = None
        self._anchor_play_after_target = None
        self.state_fetcher = StateFetcher(self)
        self._game_name = None
        self._timer_id = None
        self.is_playing = False
        self.is_strong_playing = False
        self.speed = float(favorites.speed_slider.value)
        favorites.timepoint_slider.bind('input', self._on_timepoint_slider_change)
        favorites.speed_slider.bind('input', self._on_speed_slider_change)
        favorites.play_pause_button.bind('click', lambda event: self.toggle_playing())
        favorites.game_selector.bind(
            'change',
            lambda event: setattr(self, 'game_name',
                                  favorites.game_selector.selectedOptions[0].attrs['value'])
        )

    #######################################################################
    ### Defining sequence operations: #####################################
    #                                                                     #

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, i):
        return self._states[i]

    def __iter__(self) -> Iterator[State]:
        return iter(self._states)

    def __setitem__(self, i: Union[int, slice], value: Union[Iterable[State], State]) -> None:
        raise NotImplementedError
        # self._states[i] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self._states[index]

    def insert(self, index: int, value: State) -> None:
        assert index == len(self._states) # Can only add items at the end
        self._states.insert(index, value)
        self._update_funky_states()
        window.update_ui()

    def extend(self, values: Iterable[State]) -> None:
        for v in values:
            self.append(v)

    #                                                                     #
    ### Finished defining sequence operations. ############################
    #######################################################################

    def _update_funky_states(self):
        assert len(self._states) - len(self._funky_states) == 1
        state = self._states[-1]
        try:
            previous_state = self._states[-2]
        except IndexError: # First state, Mazal Tov.
            previous_state = state

        ############################################################################################
        ### Processing players: ####################################################################
        #                                                                                          #
        players = []
        color_map = {
            -10: (0x99, 0x44, 0x44),
            -5: (0x88, 0x88, 0x22),
            0: (0x55, 0x55, 0x55),
            -1: (0x55, 0x55, 0x55),
            10: (0x55, 0x55, 0x55),
        }
        def _make_players_fluff(state):
            return {
                letter: ((player_x, player_y), color_map[reward]) for
                ((player_x, player_y), score, reward, letter, last_action_name) in state['players']
            }
        players_fluff = _make_players_fluff(state)
        previous_players_fluff = _make_players_fluff(previous_state)

        players_fluff_set = set(players_fluff)
        previous_players_fluff_set = set(previous_players_fluff)

        created_players_set = players_fluff_set - previous_players_fluff_set
        deleted_players_set = previous_players_fluff_set - players_fluff_set
        remaining_players_set = previous_players_fluff_set & players_fluff_set

        for created_player in created_players_set:
            ((player_x, player_y), color) = players_fluff[created_player]
            players.append((
                created_player, ((player_x, player_y), (player_x, player_y)),
                (color, color), (0, 1),
            ))

        for deleted_player in deleted_players_set:
            ((player_x, player_y), color) = previous_players_fluff[created_player]
            players.append((
                deleted_player, ((player_x, player_y), (player_x, player_y)),
                (color, color), (1, 0),
            ))

        for remaining_player in remaining_players_set:
            ((previous_player_x, previous_player_y), previous_color) = \
                                                            previous_players_fluff[remaining_player]
            ((player_x, player_y), color) = players_fluff[remaining_player]
            players.append((
                remaining_player, ((previous_player_x, previous_player_y), (player_x, player_y)),
                (previous_color, color), (1, 1),
            ))
        #                                                                                          #
        ### Finished processing players. ###########################################################
        ############################################################################################

        ############################################################################################
        ### Processing food: #######################################################################
        #                                                                                          #
        food = []
        old_food_positions = set(map(tuple, previous_state['food_positions']))
        new_food_positions = set(map(tuple, state['food_positions']))

        created_food_positions = new_food_positions - old_food_positions
        deleted_food_positions = old_food_positions - new_food_positions
        remaining_food_positions = old_food_positions & new_food_positions

        for created_food_position in created_food_positions:
            food.append((created_food_position, (0, 1)))

        for deleted_food_position in deleted_food_positions:
            food.append((deleted_food_position, (1, 0)))

        for remaining_food_position in remaining_food_positions:
            food.append((remaining_food_position, (1, 1)))

        #                                                                                          #
        ### Finished processing food. ##############################################################
        ############################################################################################

        ############################################################################################
        ### Processing bullets: ####################################################################
        #                                                                                          #
        bullets = []
        old_bullets = set(map(lambda x: tuple(map(tuple, x)), previous_state['bullets']))
        new_bullets = set(map(lambda x: tuple(map(tuple, x)), state['bullets']))

        while old_bullets:
            old_bullet_position, old_bullet_direction = old_bullets.pop()
            new_bullet_position_candidate = (old_bullet_position[0] + old_bullet_direction[0],
                                             old_bullet_position[1] + old_bullet_direction[1])
            new_bullet_candidate = (new_bullet_position_candidate, old_bullet_direction)
            try:
                new_bullets.remove(new_bullet_candidate)
            except KeyError:
                bullet_hit = True
            else:
                bullet_hit = False

            bullets.append((
                (old_bullet_position, new_bullet_position_candidate),
                math.tau * _directions.index(old_bullet_direction) / 4,
                (1, 1 - bullet_hit)
            ))

        for new_bullet_position, new_bullet_direction in new_bullets:
            new_bullet_previous_position = (new_bullet_position[0] - new_bullet_direction[0],
                                            new_bullet_position[1] - new_bullet_direction[1])
            bullets.append((
                (new_bullet_previous_position, new_bullet_position),
                math.tau * _directions.index(new_bullet_direction) / 4,
                (0, 1)
            ))

        #                                                                                          #
        ### Finished processing bullets. ###########################################################
        ############################################################################################

        funky_state = {
            'players': tuple(sorted(players)),
            'food': tuple(sorted(food)),
            'bullets': tuple(sorted(bullets)),
        }
        self._funky_states.append(funky_state)



        # for (player_x, player_y), score, reward, letter, last_action_name in state['players']:
            # player = favorites.get_player(letter)
            # set_new_classes(player, 'player', f'player-{letter}', f'grid-column-{player_x + 1}',
                            # f'grid-row-{player_y + 1}')
            # if reward == -5:
                # player.classList.add('player-collision')
            # else:
                # player.classList.remove('player-collision')

            # if reward == -10:
                # player.classList.add('player-shot')
            # else:
                # player.classList.remove('player-shot')

            # if score > 25:
                # player.style.opacity = '1'
            # elif score >= 1:
                # player.style.opacity = str(0.25 + (0.75 / 24) * (score - 1))
            # else:
                # player.style.opacity = '0.25'

            # player.attrs['last_action'] = str(last_action_name)






    @property
    def needle(self) -> float:
        return self._needle

    @needle.setter
    def needle(self, needle: float) -> None:
        self._needle = self._clamp_needle(needle)
        self._set_anchor()

    def _clamp_needle(self, needle):
        return clamp(round(needle), 0, max((len(self) - 1, 0)))


    def _on_timepoint_slider_change(self, event):
        self.needle = int(favorites.timepoint_slider.value)

    def _on_speed_slider_change(self, event):
        self.speed = float(favorites.speed_slider.value)
        self._set_anchor()

    def get_active(self) -> State:
        return self._states[math.ceil(self.needle)]

    def _set_anchor(self):
        self._anchor_time = time.time()
        self._anchor_needle = self.needle
        self._anchor_speed = self.speed
        self._anchor_target = False
        self._anchor_play_after_target = False


    def play(self, *, change_icon: bool = True) -> None:
        if not self.is_strong_playing:
            self._set_anchor()
            if self.needle == len(self) - 1:
                self.needle = 0
            self.is_playing = self.is_strong_playing = True
            if change_icon:
                favorites.play_pause_button_img.attrs['src'] = 'pause.png'


    def pause(self, *, change_icon: bool = True) -> None:
        self.is_playing = self.is_strong_playing = False
        if change_icon:
            favorites.play_pause_button_img.attrs['src'] = 'play.png'
        self.needle = round(self.needle + 0.2)

    def toggle_playing(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()


    def skip(self, delta: numbers.Real) -> None:
        was_strong_playing = self.is_strong_playing

        target_needle = self._clamp_needle(self.needle + delta)

        distance_to_cover = target_needle - self.needle
        time_to_cover_distance = clamp(0.1 + ((abs(distance_to_cover) - 1) / 10),
                                       minimum=0.1, maximum=0.75)

        self._anchor_time = time.time()
        self._anchor_needle = self.needle
        self._anchor_speed = distance_to_cover / time_to_cover_distance
        self._anchor_target = target_needle
        self._anchor_play_after_target = was_strong_playing
        self.is_playing = True


    @property
    def game_name(self) -> str:
        return self._game_name

    @game_name.setter
    def game_name(self, new_game_name: str) -> None:
        self.pause()
        self._game_name = new_game_name
        self.clear()
        self._funky_states.clear()
        self.needle = 0
        self.state_fetcher.update()

    def change_speed(self, delta: numbers.Real) -> None:
        self.speed = clamp(self.speed + delta, 0.2, 8)
        window.update_ui()
        self._set_anchor()



class StateFetcher:
    delay = 10_000
    def __init__(self, timeline: Timeline) -> None:
        self.timeline = timeline
        self._timer_id = None
        self.game_name = None
        self.current_batch = 0

    def update(self):
        game_name = self.timeline.game_name
        if game_name != self.game_name:
            self.current_batch = 0
            self.game_name = game_name
        timer.clear_timeout(self._timer_id)
        self._do_next()

    def _do_next(self):
        def _complete(request):
            if request.status == 200:
                self.timeline.extend(json.loads(request.responseText))
                self.current_batch += 1
                self._timer_id = timer.set_timeout(self._do_next, 0)
            elif request.status == 404:
                # self._timer_id = timer.set_timeout(self._do_next, self.delay)
                pass
            else:
                raise NotImplementedError

        url = f'games/{self.game_name}/{self.current_batch:06d}.json'
        cool_ajax(url, _complete)



timeline = window.timeline = Timeline()

key_bindings = {
    'p': timeline.toggle_playing,

    'g': lambda: timeline.skip(- float('inf')),
    'h': lambda: timeline.skip(-5),
    'j': lambda: timeline.skip(-1),
    'k': lambda: timeline.skip(1),
    'l': lambda: timeline.skip(5),
    ';': lambda: timeline.skip(len(timeline)), #float('inf')),

    'i': lambda: timeline.change_speed(-0.1),
    'o': lambda: timeline.change_speed(0.1),

    'u': update_button_handler,
}

def keypress_handler(event):
    key = event.key.lower()
    try:
        function = key_bindings[key]
    except KeyError:
        pass
    else:
        function()

def add_parameters_to_url(url: str, parameters: dict) -> str:
    # Todo: This is a caveman implementation, replace with urllib.parse
    if '?' in url:
        raise NotImplementedError
    return f'{url}?{"&".join(f"{key}={value}" for key, value in parameters.items())}'



def cool_ajax(url: str, handler: Callable, method: str = 'GET', disable_cache: bool = True) -> None:
    request = ajax.ajax()
    if disable_cache:
        url = add_parameters_to_url(url, {'_': random.randint(0, 10**8)})
    request.open(method, url, True)
    request.bind('complete', handler)
    request.send()

def update_game_names(*, loop: bool = False, select_newest: bool = False) -> None:
    def _complete(request):
        game_names = re.findall('a href="([^/]+)/"', request.responseText)
        for game_name in game_names:
            if not document.select(f'#game-selector option[value="{game_name}"]'):
                option = html.OPTION(value=game_name)
                option.textContent = game_name
                favorites.game_selector <= option
        if select_newest:
            latest_game_name = get_latest_game_name()
            latest_game_name_option = document.select(
                                      f'#game-selector option[value="{get_latest_game_name()}"]')[0]
            latest_game_name_option.selected = True
            timeline.game_name = latest_game_name
        if loop:
            timer.set_timeout(lambda: update_game_names(loop=True), 10_000)
    cool_ajax('games/', _complete)

def get_latest_game_name():
    return max(option.attrs['value'] for option in document.select(f'#game-selector option'))


# def build_player_avatar(letter: str):
    # player_avatar = pixi.Container.new()
    # window.app.stage.addChild(player_avatar)

    # # circle = Sprite.new(resources['images/circle.png'].texture)
    # graphic = pixi.Graphics.new()
    # graphic.beginFill(0x555555)
    # graphic.drawCircle(0, 0, HALF_CELL_SIZE - 1)

    # # circle.anchor.set(0.5)
    # # player_avatar.addChild(circle)
    # graphic.pivot.set(0.5)
    # player_avatar.addChild(graphic)

    # letter_avatar = pixi.Text.new(
        # letter,
        # {
            # 'fontFamily': 'Consolas',
            # 'fontSize': CELL_SIZE - 6,
            # 'fill': 'white',
            # 'stroke': '#ff3300',
            # 'fontWeight': 'bold',
            # # 'strokeThickness': 4,
            # # 'dropShadow': true,
            # # 'dropShadowColor': '#000000',
            # # 'dropShadowBlur': 4,
            # # 'dropShadowAngle': Math.PI / 6,
            # # 'dropShadowDistance': 6,
        # }
    # )
    # letter_avatar.anchor.set(0.5)
    # player_avatar.addChild(letter_avatar)

    # # player_avatar.pivot.x = 0.5 * (player_avatar.width / player_avatar.scale.x)
    # # player_avatar.pivot.y = 0.5 * (player_avatar.height / player_avatar.scale.y)

    # return player_avatar

def main():
    document.bind('keypress', keypress_handler)
    update_game_names(loop=False, select_newest=True)
    window._animate_lock = False
    favorites.update_button.bind('click', update_button_handler)

    timer.request_animation_frame(window.animate)


main()