import game
import numpy as np
from scipy.optimize import minimize
import math


def rate_players(players, n_games_per_pair, elo_scale=math.log(10) / 400):
    win_matrix = np.zeros([len(players), len(players)])
    for i1, player1 in enumerate(players):
        for i2, player2 in enumerate(players):
            if i1 == i2:
                continue
            for _ in range(n_games_per_pair):
                outcome = game.match(A=player1, B=player2)
                if outcome == 'WIN_A':
                    win_matrix[i1, i2] += 1
                elif outcome == 'WIN_B':
                    win_matrix[i2, i1] += 1
    result = minimize(elo_from_win_matrix,
                      x0=np.zeros(len(players) - 1),
                      args=(win_matrix, elo_scale))
    return [0] + result['x'].tolist()


def elo_from_win_matrix(x0, win_matrix, elo_scale):
    strength = np.concatenate([[0], x0 * elo_scale])
    win_probs = 1 / (1 + np.exp(-np.subtract.outer(strength, strength)))
    return -np.sum(win_matrix * np.log(win_probs))


def rate_verbose(player1, player2, n_games, elo_scale=math.log(10) / 400):
    combined_results = [0, 0, 0]  # win, lose, tie
    print('            WIN   LOSE    TIE    ELO')
    for order in range(2):
        players = dict(A=player1, B=player2) if order == 0 else dict(A=player2, B=player1)
        name = players['A'].evaluator_name
        results = [0, 0, 0]  # win, lost, tie
        for i in range(n_games):
            print('\rA={:6s}  {:5d}  {:5d}  {:5d}'.format(name, *results), end='', flush=True)
            outcome = game.match(**players)
            if outcome == 'WIN_A':
                results[order] += 1
                combined_results[order] += 1
            elif outcome == 'WIN_B':
                results[1 - order] += 1
                combined_results[1 - order] += 1
            else:
                results[2] += 1
                combined_results[2] += 1
        print('\rA={:6s}  {:5d}  {:5d}  {:5d}'.format(name, *results))
    approx_elo = math.log(combined_results[0] / combined_results[1]) / elo_scale
    print('total     {:5d}  {:5d}  {:5d}  {:5d}'.format(*(combined_results + [int(approx_elo)])))
    return approx_elo
