import game
from game import RandomPlayer
from mcts import MCTS_Player, Uninformative
from collections import Counter
import rating


# game.match(A=RandomPlayer(), B=MCTS_Player(Uninformative(), 200)).print()
# print()
# game.match(A=MCTS_Player(Uninformative(), 200), B=MCTS_Player(Uninformative(), 200)).print()
# print()


# print(Counter(game.match(A=MCTS_Player(Uninformative(), 20), B=RandomPlayer()).outcome for i in range(100)))
# print(Counter(game.match(B=MCTS_Player(Uninformative(), 20), A=RandomPlayer()).outcome for i in range(100)))
# print(Counter(game.match(B=MCTS_Player(Uninformative(), 50), A=RandomPlayer()).outcome for i in range(100)))
# print(Counter(game.match(B=MCTS_Player(Uninformative(), 200), A=RandomPlayer()).outcome for i in range(100)))
# print(Counter(game.match(A=MCTS_Player(Uninformative(), 10), B=MCTS_Player(Uninformative(), 10)).outcome for i in range(100)))


def mcts_factory(n_playouts):
    return lambda: MCTS_Player(Uninformative(), n_playouts=n_playouts)
playout_values = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 100, 150, 200]
ratings = rating.rate_players([RandomPlayer] + list(map(mcts_factory, playout_values)), 10)
for n_playouts, elo in zip(playout_values, ratings[1:]):
    print('Rating for {:3d} random playouts: {:4d}'.format(n_playouts, int(elo)))

print()
print('States explored: {}'.format(len(game.State._actions)))
