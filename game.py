import random
from collections import namedtuple


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

NEXT_PLAYER = {'A': 'B', 'B': 'A'}

# game-theoretic values for player A
OUTCOMES = {
    'WIN_A': 1,
    'WIN_B': -1,
    'TIE': 0
}


class State(object):
    INITIAL = ('.' * (ROWS * COLUMNS), 'A')
    _actions = {}

    @classmethod
    def actions(cls, state):
        if state in cls._actions:
            return cls._actions[state]
        board, player = state
        other = NEXT_PLAYER[player]
        actions = {}
        for action in range(COLUMNS):
            board_index = cls._board_index(board, action)
            if board_index is not None:
                next_board = board[:board_index] + player + board[board_index+1:]
                maybe_final_score = cls._score_position(next_board, board_index, player)
                next_state = maybe_final_score or (next_board, other)
                actions[action] = next_state
        assert len(actions) > 0
        cls._actions[state] = actions
        return actions

    @classmethod
    def _board_index(cls, board, action):
        index = board.find('.', action * ROWS)
        if 0 <= index - action * ROWS < ROWS:
            return index
        else:
            return None

    @classmethod
    def _score_position(cls, board, last_move, last_player):
        for line in LINES_BY_INDEX[last_move]:
            if all(board[i] == last_player for i in line):
                return 'WIN_' + last_player
        if '.' not in board:
            return 'TIE'
        return None


class Player(object):
    def choose_action(self, state):
        raise NotImplementedError()


class RandomPlayer(Player):
    def choose_action(self, state):
        return random.choice(list(State.actions(state).keys()))


class GameLog(object):
    "Lists of all the actions and responses in the game"
    StateAction = namedtuple('StateAction', ['state', 'action'])

    def __init__(self):
        self._journal = []
        self.outcome = None

    def record_move(self, state, action):
        self._journal.append(self.StateAction(state=state, action=action))

    def record_outcome(self, outcome):
        self.outcome = outcome

    def print(self):
        last_board, last_player = self._journal[-1].state
        last_action = self._journal[-1].action
        trace = [[] for i in range(COLUMNS)]
        for i, event in enumerate(self._journal):
            trace[event.action].append(i)
        print('{board}\n\n{player} plays {action}\n\n{trace}\n\nGame outcome: {outcome}'.format(
            board=self._str_board(last_board),
            player=last_player,
            action=last_action,
            trace=self._str_trace(trace),
            outcome=self.outcome))

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

    def positions_dict(self):
        value = OUTCOMES[self.outcome]
        positions = {'A': [], 'B': []}
        for event in self._journal:
            _, player = event.state
            positions[player].append((event.state, event.action, value))
        return positions


def match(**players):
    log = GameLog()
    state = State.INITIAL
    while state not in OUTCOMES:
        _, player = state
        action = players[player].choose_action(state)
        log.record_move(state, action)
        state = State.actions(state)[action]
    log.record_outcome(state)
    return log
