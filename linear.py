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

    def evaluate(self, state):
        if random.random() < 0.5:
            state = game.State.random_permute(state)
        action_probs = self.session.run(self.action_probs, feed_dict={
            self.features: [game.GameLog._encode(state)]
        })[0]
        return action_probs, 0

    def train(self, batch_states, batch_probs, n_steps):
        logging.info('Gradient descent, 1000 iterations')
        feed_dict = {
            self.features: batch_states,
            self.chosen_action: batch_probs
        }
        loss = []
        for i in Progress(range(n_steps)):
            if i % 100 == 0:
                loss.append(self.session.run(self.cross_entropy, feed_dict=feed_dict))
            self.session.run(self.train_op, feed_dict=feed_dict)
        loss.append(self.session.run(self.cross_entropy, feed_dict=feed_dict))
        logging.info('Training losses: {}'.format(', '.join('{:.3f}'.format(x) for x in loss)))

    def compare_with_naive(self):
        total_win, total_lose, total_tie = 0, 0, 0
        print('           WIN   LOSE    TIE    ELO')
        win, lose, tie = 0, 0, 0
        print('A=smart  {:5d}  {:5d}  {:5d}'.format(win, lose, tie), end='', flush=True)
        for i in range(50):
            outcome = game.match(A=mcts.MCTS_Player(self, 30), B=mcts.MCTS_Player(mcts.Uninformative(), 30)).outcome
            if outcome == 'WIN_A':
                win += 1
                total_win += 1
            elif outcome == 'WIN_B':
                lose += 1
                total_lose += 1
            else:
                tie += 1
                total_tie += 1
            print('\rA=smart  {:5d}  {:5d}  {:5d}'.format(win, lose, tie), end='', flush=True)
        print()
        win, lose, tie = 0, 0, 0
        print('A=naive  {:5d}  {:5d}  {:5d}'.format(win, lose, tie), end='', flush=True)
        for i in range(50):
            outcome = game.match(A=mcts.MCTS_Player(mcts.Uninformative(), 30), B=mcts.MCTS_Player(self, 30)).outcome
            if outcome == 'WIN_B':
                win += 1
                total_win += 1
            elif outcome == 'WIN_A':
                lose += 1
                total_lose += 1
            else:
                tie += 1
                total_tie += 1
            print('\rA=naive  {:5d}  {:5d}  {:5d}'.format(win, lose, tie), end='', flush=True)
        print()
        approx_elo = int(400 * math.log(total_win / total_lose))
        print('total    {:5d}  {:5d}  {:5d}  {:5d}'.format(total_win, total_lose, total_tie, approx_elo))
        print()


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]  %(message)s')


lin_eval = LinearEvaluator()

logging.info('Initial performance:')
lin_eval.compare_with_naive()

for i in range(20):
    logging.info('Running 500 self-play games')
    batch_states = []
    batch_probs = []
    for j in Progress(range(500)):
        results = game.match(A=mcts.MCTS_Player(lin_eval, 30),
                             B=mcts.MCTS_Player(lin_eval, 30))
        states, probs, _ = results.training_vectors()
        batch_states.extend(states)
        batch_probs.extend(probs)
    lin_eval.train(batch_states, batch_probs, 1000)
    logging.info('Performance:')
    lin_eval.compare_with_naive()
