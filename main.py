import game
import random


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


def play_random_game():
    trace = [[] for i in range(game.COLUMNS)]
    state = game.State.INITIAL
    i = 1
    while state not in game.OUTCOMES:
        action, next_state = random.choice(game.State.actions(state))
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


play_random_game()


# for line in game.LINES:
#     pieces = ['.'] * game.ROWS * game.COLUMNS
#     for i in line:
#         pieces[i] = 'X'
#     print_board(''.join(pieces))
#     print()
#     print()
