#!python

import json
import io
import sys
import itertools
import pathlib

states = json.loads(pathlib.Path(r'C:\Users\Administrator\Documents\Python '
                                 r'projects\simulate_society\pal\game\000003.json').read_text())

state_number = int(sys.argv[1])

state = states[state_number]

players = {
    (player_x, player_y): (score, letter)
    for (player_x, player_y), score, letter in state['players']
}

food_positions = set(map(tuple, state['food_positions']))

string_io = io.StringIO()
for x, y in itertools.product(range(24), repeat=2):
    if x != 0 and y == 0:
        string_io.write('|\n')
    if (x, y) in players:
        string_io.write(players[(x, y)][1])
    elif (x, y) in food_positions:
        string_io.write('+')
    else:
        string_io.write(' ')

string_io.write('|\n')
string_io.write('‾' * 24)
string_io.write('\n')

print(string_io.getvalue())