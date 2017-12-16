import tensorflow as tf
import game
import mcts
import random
import math
import logging


class Progress(object):
    def __init__(self, iterable):
        self._length = len(iterable)
        self._iterator = iter(iterable)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            value = next(self._iterator)
            print('\r{} / {}'.format(self._index, self._length), end='', flush=True)
            self._index += 1
            return value
        except StopIteration:
            print('\r', end='', flush=True)
            raise


class LinearEvaluator(mcts.Evaluator):
    N_FEATURES = 2
    INPUT_SIZE = game.ROWS * game.COLUMNS * N_FEATURES

    def __init__(self):
        self.features = tf.placeholder(tf.float32, [None, self.N_FEATURES, game.ROWS * game.COLUMNS])
        self.chosen_action = tf.placeholder(tf.float32, [None, game.COLUMNS])
        self.matrix = tf.Variable(tf.zeros([self.INPUT_SIZE, game.COLUMNS]))
        self.affine = tf.Variable(tf.zeros([game.COLUMNS]))
        flat_features = tf.reshape(self.features, [-1, self.INPUT_SIZE])
        action_logits = tf.matmul(flat_features, self.matrix) + self.affine
        self.action_probs = tf.nn.softmax(action_logits)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.chosen_action, logits=action_logits))
        self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    @classmethod
    def initial(cls):
        return cls()

    def evaluate(self, features):
        if random.random() < 0.5:
            features = game.reflect_columns(features)
        action_probs = self.session.run(self.action_probs, feed_dict={
            self.features: [features]
        })[0]
        return action_probs, 0

    @classmethod
    def train(cls, batch_states, batch_probs, mini_batch_size, n_steps):
        obj = cls()
        obj._train(batch_states, batch_probs, mini_batch_size, n_steps)
        return obj

    def _train(self, batch_states, batch_probs, mini_batch_size, n_steps):
        batch_size = len(batch_states)
        logging.info('Gradient descent with batch size {}'.format(batch_size))
        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(batch_states),
            tf.data.Dataset.from_tensor_slices(batch_probs)))
        dataset = dataset.shuffle(2**16).repeat().batch(mini_batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
        validation_inds = random.sample(range(batch_size), min(batch_size, 100))
        validation_feed_dict = {
            self.features: [batch_states[i] for i in validation_inds],
            self.chosen_action: [batch_probs[i] for i in validation_inds]}
        logging.info('Performing {} iterations with mini-batch size {}'.format(n_steps, mini_batch_size))
        loss = []
        for i in range(n_steps):
            next_states, next_probs = self.session.run(next_batch)
            feed_dict = {
                self.features: next_states,
                self.chosen_action: next_probs
            }
            if i % 100 == 0:
                loss.append(self.session.run(self.cross_entropy, feed_dict=validation_feed_dict))
            self.session.run(self.train_op, feed_dict=feed_dict)
        loss.append(self.session.run(self.cross_entropy, feed_dict=validation_feed_dict))
        logging.info('Training losses: {}'.format(', '.join('{:.3f}'.format(x) for x in loss)))


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]  %(message)s')


def compare_evaluators(evaluators, n_games, n_playouts, verbose):
    "Evaluators is list of (name, evaluator)"
    combined_results = [0, 0, 0]  # win, lose, tie
    if verbose:
        print('            WIN   LOSE    TIE    ELO')
    for order in range(2):
        results = [0, 0, 0]  # win, lost, tie
        for i in range(n_games):
            [(name, eval_A), (_, eval_B)] = evaluators[order], evaluators[1 - order]
            if verbose:
                print('\rA={:6s}  {:5d}  {:5d}  {:5d}'.format(name, *results), end='', flush=True)
            outcome = game.match(A=mcts.MCTS_Player(eval_A, n_playouts), B=mcts.MCTS_Player(eval_B, n_playouts))
            if outcome == 'WIN_A':
                results[order] += 1
                combined_results[order] += 1
            elif outcome == 'WIN_B':
                results[1 - order] += 1
                combined_results[1 - order] += 1
            else:
                results[2] += 1
                combined_results[2] += 1
        if verbose:
            print('\rA={:6s}  {:5d}  {:5d}  {:5d}'.format(name, *results))
    approx_elo = int(400 * math.log(combined_results[0] / combined_results[1], 10))
    if verbose:
        print('total     {:5d}  {:5d}  {:5d}  {:5d}'.format(*(combined_results + [approx_elo])))
    return approx_elo


linear_evaluators = [LinearEvaluator.initial()]
current_best = 0

named_naive = ('naive', mcts.Uninformative())
named_linear = lambda i: ('lin_{}'.format(i), linear_evaluators[i])

try:
    all_states = []
    all_probs = []
    for i in range(20):
        logging.info('Running 500 self-play games')
        for j in Progress(range(500)):
            manager = game.GameManager({
                'A': mcts.MCTS_Player(linear_evaluators[current_best], 30),
                'B': mcts.MCTS_Player(linear_evaluators[current_best], 30)
            })
            manager.run()
            all_states.extend(manager.encoded_boards)
            all_probs.extend(manager.prob_vecs)
        linear_evaluators.append(LinearEvaluator.train(all_states, all_probs, 2048, 1000))
        logging.info('Performance against previous:')
        score = compare_evaluators(
            [named_linear(len(linear_evaluators) - 1), named_linear(current_best)],
            n_games=50, n_playouts=30, verbose=True)
        if score > 34.8:
            logging.info('Replacing best evaluator')
            linear_evaluators[current_best] = None
            current_best = len(linear_evaluators) - 1
            logging.info('Performance against naive:')
            score = compare_evaluators(
                [named_linear(current_best), named_naive],
                n_games=50, n_playouts=30, verbose=True)
        else:
            linear_evaluators[-1] = None
        print()
except KeyboardInterrupt:
    print()
    logging.info('Interrupted, leaving')
