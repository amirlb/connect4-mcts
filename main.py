import game
import random
import mcts
import numpy as np
from collections import Counter


class Uniform(mcts.Evaluator):
    def evaluate(self, state):
        return np.ones(game.COLUMNS) / game.COLUMNS, 0


class Player(object):
    def choose_action(self, state):
        raise NotImplementedError()

class RandomPlayer(Player):
    def choose_action(self, state):
        return random.choice(list(game.State.actions(state).keys()))

class MCTSPlayer(Player):
    def __init__(self, evaluator, n_playouts):
        self._moves_graph = mcts.MovesGraph(evaluator)
        self._n_playouts = n_playouts
    def choose_action(self, state):
        return self._moves_graph.choose_action(state, self._n_playouts)


def print_board(board):
    for y in reversed(range(game.ROWS)):
        for x in range(game.COLUMNS):
            index = x * game.ROWS + y
            print(' '+board[index]+' ', end='')
        print()


def print_trace(trace):
    for y in reversed(range(game.ROWS)):
        for x in range(game.COLUMNS):
            print('{:2}'.format(trace[x][y] or ' .'), end=' ')
        print()


def show_computer_game(players):
    trace = [[] for i in range(game.COLUMNS)]
    state = game.State.INITIAL
    i = 1
    while state not in game.OUTCOMES:
        _, player = state
        action = players[player].choose_action(state)
        next_state = game.State.actions(state)[action]
        if next_state in game.OUTCOMES:
            board, player = state
            print_board(board)
            print()
            print('{} plays {}'.format(player, action))
            print()
        trace[action].append(i)
        i += 1
        state = next_state
    trace = [col + [None]*(7 - len(col)) for col in trace]
    print_trace(trace)
    print()
    print('Game outcome: ' + state)


# show_computer_game({'A': RandomPlayer(), 'B': MCTSPlayer(Uniform(), 200)})


def match_result(players):
    state = game.State.INITIAL
    while state not in game.OUTCOMES:
        # print(repr(state))
        _, player = state
        # print(repr(player))
        # print()
        action = players[player].choose_action(state)
        state = game.State.actions(state)[action]
    return state

print(Counter(match_result({'A': MCTSPlayer(Uniform(), 20), 'B': RandomPlayer()}) for i in range(100)))
print(Counter(match_result({'B': MCTSPlayer(Uniform(), 20), 'A': RandomPlayer()}) for i in range(100)))
print(Counter(match_result({'B': MCTSPlayer(Uniform(), 50), 'A': RandomPlayer()}) for i in range(100)))
print(Counter(match_result({'B': MCTSPlayer(Uniform(), 200), 'A': RandomPlayer()}) for i in range(100)))
print(Counter(match_result({'A': MCTSPlayer(Uniform(), 10), 'B': MCTSPlayer(Uniform(), 10)}) for i in range(100)))
