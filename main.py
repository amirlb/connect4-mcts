import game
from game import RandomPlayer
from mcts import MCTS_Player, Uninformative
from collections import Counter
import rating


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


# show_computer_game({'A': RandomPlayer(), 'B': MCTS_Player(Uninformative(), 200)})


# print(Counter(game.match_result({'A': MCTS_Player(Uninformative(), 20), 'B': RandomPlayer()}) for i in range(100)))
# print(Counter(game.match_result({'B': MCTS_Player(Uninformative(), 20), 'A': RandomPlayer()}) for i in range(100)))
# print(Counter(match_result({'B': MCTS_Player(Uninformative(), 50), 'A': RandomPlayer()}) for i in range(100)))
# print(Counter(match_result({'B': MCTS_Player(Uninformative(), 200), 'A': RandomPlayer()}) for i in range(100)))
# print(Counter(game.match_result({'A': MCTS_Player(Uninformative(), 10), 'B': MCTS_Player(Uninformative(), 10)}) for i in range(100)))


mcts_0 = lambda: MCTS_Player(Uninformative(), 0)
mcts_10 = lambda: MCTS_Player(Uninformative(), 10)
mcts_20 = lambda: MCTS_Player(Uninformative(), 20)
mcts_50 = lambda: MCTS_Player(Uninformative(), 50)
mcts_100 = lambda: MCTS_Player(Uninformative(), 100)
ratings = rating.rate_players([RandomPlayer, mcts_0, mcts_10, mcts_20, mcts_50, mcts_100], 10)
for n_playouts, elo in zip([0, 10, 20, 50], ratings[1:]):
    print('Rating for {:3d} random playouts: {:4d}'.format(n_playouts, int(elo)))
