from util import Progress
import mcts
import logging
import game
import rating


class LearningEvaluator(mcts.Evaluator):
    def train(self, states, probs):
        "Create a new copy of self and train its parameters using the given states and action probs"
        raise NotImplementedError


def train_best(learning_evaluator, n_epochs, n_train_games, n_train_playouts, n_compare_games, n_compare_playouts):
    try:
        logging.info('Initial performance:')
        rating.rate_verbose(
            mcts.MCTS_Player(learning_evaluator, n_compare_playouts),
            mcts.MCTS_Player(mcts.Uninformative(), n_compare_playouts),
            n_games=n_compare_games)
        print()
        all_states = []
        all_probs = []
        for i in range(n_epochs):
            logging.info('Running {} self-play games'.format(n_train_games))
            for j in Progress(range(n_train_games)):
                manager = game.GameManager({
                    'A': mcts.MCTS_Player(learning_evaluator, n_train_playouts),
                    'B': mcts.MCTS_Player(learning_evaluator, n_train_playouts)
                })
                manager.run()
                all_states.extend(manager.encoded_boards)
                all_probs.extend(manager.prob_vecs)
            next_gen = learning_evaluator.train(all_states, all_probs)
            logging.info('Performance against previous:')
            score = rating.rate_verbose(
                mcts.MCTS_Player(next_gen, n_compare_playouts),
                mcts.MCTS_Player(learning_evaluator, n_compare_playouts),
                n_games=n_compare_games)
            if score > 34.8:
                # 55% win rate
                logging.info('Replacing best evaluator')
                learning_evaluator = next_gen
                logging.info('Performance against naive:')
                rating.rate_verbose(
                    mcts.MCTS_Player(learning_evaluator, n_compare_playouts),
                    mcts.MCTS_Player(mcts.Uninformative(), n_compare_playouts),
                    n_games=n_compare_games)
            print()
    except KeyboardInterrupt:
        print()
        logging.info('Interrupted, leaving')

    return learning_evaluator