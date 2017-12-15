import game
import numpy as np
from scipy.optimize import minimize


def rate_players(players, n_games_per_pair, elo_scale=1/400):
    win_matrix = np.zeros([len(players), len(players)])
    for i1, player1 in enumerate(players):
        for i2, player2 in enumerate(players[:i1]):
            for j in range(n_games_per_pair):
                outcome = game.match(A=player1(), B=player2()).outcome
                if outcome == 'WIN_A':
                    win_matrix[i1, i2] += 1
                elif outcome == 'WIN_B':
                    win_matrix[i2, i1] += 1
                outcome = game.match(A=player2(), B=player1()).outcome
                if outcome == 'WIN_A':
                    win_matrix[i2, i1] += 1
                elif outcome == 'WIN_B':
                    win_matrix[i1, i2] += 1
    result = minimize(elo_from_win_matrix,
                      x0=np.zeros(len(players) - 1),
                      args=(win_matrix, elo_scale))
    return [0] + result['x'].tolist()


def elo_from_win_matrix(x0, win_matrix, elo_scale):
    strength = np.concatenate([[0], x0 * elo_scale])
    win_probs = 1 / (1 + np.exp(-np.subtract.outer(strength, strength)))
    return -np.sum(win_matrix * np.log(win_probs))
