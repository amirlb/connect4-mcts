import random
from functools import lru_cache


COLUMNS = 7
ROWS = 6

LINES = []
# vertical
for x in range(COLUMNS):
    for y in range(ROWS - 3):
        LINES.append([x*ROWS + (y+i) for i in range(4)])
# horizontal
for y in range(ROWS):
    for x in range(COLUMNS - 3):
        LINES.append([(x+i)*ROWS + y for i in range(4)])
# diagonal
for x in range(COLUMNS - 3):
    for y in range(ROWS - 3):
        LINES.append([(x+i)*ROWS + (y+i) for i in range(4)])
    for y in range(3, ROWS):
        LINES.append([(x+i)*ROWS + (y-i) for i in range(4)])

LINES_BY_INDEX = {i: [] for i in range(COLUMNS * ROWS)}
for line in LINES:
    for i in line:
        LINES_BY_INDEX[i].append(line)

INITIAL_STATE = ('.' * (ROWS * COLUMNS), 'A')  # empty board, A to play

NEXT_PLAYER = {'A': 'B', 'B': 'A'}

# game-theoretic values for player A
OUTCOMES = {
    'WIN_A': 1,
    'WIN_B': -1,
    'TIE': 0
}


def _score_position(board, last_move, last_player):
    for line in LINES_BY_INDEX[last_move]:
        if all(board[i] == last_player for i in line):
            return 'WIN_' + last_player
    if '.' not in board:
        return 'TIE'
    return None


@lru_cache(maxsize=2**19)
def actions(state):
    board, player = state
    other = NEXT_PLAYER[player]
    possibilities = {}
    for action in range(COLUMNS):
        board_index = board.find('.', action * ROWS)
        if 0 <= board_index - action * ROWS < ROWS:
            next_board = board[:board_index] + player + board[board_index+1:]
            maybe_final_score = _score_position(next_board, board_index, player)
            next_state = maybe_final_score or (next_board, other)
            possibilities[action] = next_state
    assert len(possibilities) > 0
    return possibilities


def encode_board(state):
    "Return features, shape [4, ROWS * COLUMNS]"
    board, player = state
    other = NEXT_PLAYER[player]
    return [[int(player == 'A')] * (ROWS * COLUMNS),
            [int(player == 'B')] * (ROWS * COLUMNS),
            [int(x == player) for x in board],
            [int(x == other) for x in board]]


def reflect_columns(features):
    return [sum((layer[i:i + ROWS] for i in reversed(range(0, ROWS*COLUMNS, ROWS))), [])
            for layer in features]


class Player(object):
    def reset(self):
        "Clear any in-game state. Called at the start of matches"
        pass

    def choose_action(self, match):
        "Returns a tuple of (policy probabilities, action)"
        raise NotImplementedError()


class RandomPlayer(Player):
    def choose_action(self, match):
        state = match.states[-1]
        return None, random.choice(list(actions(state).keys()))


class GameManager(object):

    def __init__(self, players):
        self.players = players
        self.current_state = INITIAL_STATE
        self.move_number = 0
        self.outcome = None
        # journal
        self.states = []
        self.encoded_boards = []
        self.prob_vecs = []
        self.actions = []

    def run(self):
        for player in self.players.values():
            player.reset()
        while self.current_state not in OUTCOMES:
            self.states.append(self.current_state)
            self.encoded_boards.append(encode_board(self.current_state))
            _, player = self.current_state
            probs, action = self.players[player].choose_action(self)
            self.prob_vecs.append(probs)
            self.actions.append(action)
            self.current_state = actions(self.current_state)[action]
            self.move_number += 1
        self.outcome = self.current_state
        self.current_state = None
        return self

    def __str__(self):
        last_board, last_player = self.states[-1]
        last_action = self.actions[-1]
        trace = [[] for i in range(COLUMNS)]
        for i, action in enumerate(self.actions):
            trace[action].append(i)
        return '{board}\n\n{player} plays {action}\n\n{trace}\n\nGame outcome: {outcome}'.format(
            board=self._str_board(last_board),
            player=last_player,
            action=last_action,
            trace=self._str_trace(trace),
            outcome=self.outcome)

    @staticmethod
    def _str_board(board):
        return '\n'.join(''.join(' {} '.format(board[x * ROWS + y])
                                 for x in range(COLUMNS))
                         for y in reversed(range(ROWS)))

    @staticmethod
    def _str_trace(trace):
        return '\n'.join(' '.join('{:2d}'.format(trace[x][y] + 1) if y < len(trace[x]) else ' .'
                                  for x in range(COLUMNS))
                         for y in reversed(range(ROWS)))


def match(**players):
    return GameManager(players).run().outcome
