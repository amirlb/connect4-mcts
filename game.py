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
                assert board[board_index] == '.'
                next_board = board[:board_index] + player + board[board_index+1:]
                maybe_final_score = cls._score_position(next_board, board_index, player)
                assert maybe_final_score is None or maybe_final_score in OUTCOMES
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
