import game
from game import RandomPlayer
from mcts import MCTS_Player, Uninformative
from collections import Counter
import rating
import learning
import linear
import logging
import all_to_all_nn


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]  %(message)s')


# print(game.GameManager({'A': RandomPlayer(), 'B': MCTS_Player(Uninformative(), 200)}).run())
# print()
# print(game.GameManager({'A': MCTS_Player(Uninformative(), 200), 'B': MCTS_Player(Uninformative(), 200)}).run())
# print()


# print(Counter(game.match(A=MCTS_Player(Uninformative(), 20), B=RandomPlayer()) for i in range(100)))
# print(Counter(game.match(B=MCTS_Player(Uninformative(), 20), A=RandomPlayer()) for i in range(100)))
# print(Counter(game.match(B=MCTS_Player(Uninformative(), 50), A=RandomPlayer()) for i in range(100)))
# print(Counter(game.match(B=MCTS_Player(Uninformative(), 200), A=RandomPlayer()) for i in range(100)))
# print(Counter(game.match(A=MCTS_Player(Uninformative(), 10), B=MCTS_Player(Uninformative(), 10)) for i in range(100)))


# playout_values = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 100, 150, 200]
# ratings = rating.rate_players(
#     [RandomPlayer()] + [MCTS_Player(Uninformative(), n_playouts=n) for n in playout_values],
#     10)
# for n_playouts, elo in zip(playout_values, ratings[1:]):
#     print('Rating for {:3d} random playouts: {:4d}'.format(n_playouts, int(elo)))


learning.train_best(
    # linear.LinearEvaluator(),
    all_to_all_nn.FullyConnectedEvaluator(n_layers=5, n_layer_neurons=128, regularization=1e-2),
    20, 500, 30, 50, 30
)
