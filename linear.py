import tensorflow as tf
import game
import mcts
import random
import time
import math


class LinearEvaluator(mcts.Evaluator):
    N_FEATURES = game.ROWS * game.COLUMNS * 2

    def __init__(self):
        self.features = tf.placeholder(tf.float32, [None, self.N_FEATURES])
        self.chosen_action = tf.placeholder(tf.float32, [None, game.COLUMNS])
        self.matrix = tf.Variable(tf.zeros([self.N_FEATURES, game.COLUMNS]))
        self.affine = tf.Variable(tf.zeros([game.COLUMNS]))
        action_logits = tf.matmul(self.features, self.matrix) + self.affine
        self.action_probs = tf.nn.softmax(action_logits)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.chosen_action, logits=action_logits))
        self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def evaluate(self, state):
        if random.random() < 0.5:
            state = self._flip_board(state)
        action_probs = self.session.run(self.action_probs, feed_dict={
            self.features: [self._encode_state(state)]
        })[0]
        return action_probs, 0

    @staticmethod
    def _flip_board(state):
        board, player = state
        board = ''.join(board[i - game.ROWS : i] for i in range(game.ROWS * game.COLUMNS, 0, -game.ROWS))
        return (board, player)

    @staticmethod
    def _encode_state(state):
        board, _ = state
        return [board[i] == p for i in range(game.ROWS * game.COLUMNS) for p in ('A', 'B')]

    @staticmethod
    def _encode_action(action):
        vec = [0] * game.COLUMNS
        vec[action] = 1
        return vec

    def train(self, batch, n_steps):
        feed_dict = {
            self.features: [self._encode_state(state) for state, action, value in batch],
            self.chosen_action: [self._encode_action(action) for state, action, value in batch]
        }
        for _ in range(n_steps):
            self.session.run(self.train_op, feed_dict=feed_dict)

    def compare_with_naive(self):
        total_win, total_lose, total_tie = 0, 0, 0
        print('           WIN   LOSE    TIE    ELO')
        win, lose, tie = 0, 0, 0
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
        print('A=smart  {:5d}  {:5d}  {:5d}'.format(win, lose, tie))
        win, lose, tie = 0, 0, 0
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
        print('A=naive  {:5d}  {:5d}  {:5d}'.format(win, lose, tie))
        approx_elo = int(400 * math.log(total_win / total_lose))
        print('total    {:5d}  {:5d}  {:5d}  {:5d}'.format(total_win, total_lose, total_tie, approx_elo))


lin_eval = LinearEvaluator()

print()
print(time.ctime(), '  initial performance:')
lin_eval.compare_with_naive()

for i in range(20):
    print()
    print(time.ctime(), '  running 500 self-play games')
    batch = []
    for j in range(500):
        results = game.match(A=mcts.MCTS_Player(lin_eval, 30),
                             B=mcts.MCTS_Player(lin_eval, 30)).positions_dict()
        for data in results.values():
            batch.extend(data)
    print(time.ctime(), '  gradient descent, 1000 iterations')
    lin_eval.train(batch, 1000)
    print(time.ctime(), '  performance:')
    lin_eval.compare_with_naive()
