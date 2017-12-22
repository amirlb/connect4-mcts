import tensorflow as tf
import game
import random
import learning
import logging


class LinearEvaluator(learning.LearningEvaluator):
    N_FEATURES = 2
    INPUT_SIZE = game.ROWS * game.COLUMNS * N_FEATURES
    MINI_BATCH_SIZE = 2048
    N_TRAIN_STEPS = 1000

    COUNTER = -1
    @classmethod
    def _increment(cls):
        cls.COUNTER += 1
        return cls.COUNTER

    def __init__(self, matrix_value=None, affine_value=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.index = self._increment()
            self.features = tf.placeholder(tf.float32, [None, self.N_FEATURES, game.ROWS * game.COLUMNS])
            self.chosen_action = tf.placeholder(tf.float32, [None, game.COLUMNS])
            self.matrix = tf.Variable(tf.zeros([self.INPUT_SIZE, game.COLUMNS]) if matrix_value is None else matrix_value)
            self.affine = tf.Variable(tf.zeros([game.COLUMNS]) if affine_value is None else affine_value)
            flat_features = tf.reshape(self.features, [-1, self.INPUT_SIZE])
            action_logits = tf.matmul(flat_features, self.matrix) + self.affine
            self.action_probs = tf.nn.softmax(action_logits)
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.chosen_action, logits=action_logits))
            self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

    def evaluate(self, features):
        with self.graph.as_default():
            if random.random() < 0.5:
                features = game.reflect_columns(features)
            action_probs = self.session.run(self.action_probs, feed_dict={
                self.features: [features]
            })[0]
            return action_probs, 0

    def train(self, batch_states, batch_probs):
        with self.graph.as_default():
            matrix_value, affine_value = self.session.run([self.matrix, self.affine])
        obj = self.__class__(matrix_value, affine_value)
        obj._train(batch_states, batch_probs)
        return obj

    def get_name(self):
        return 'lin_{}'.format(self.index)

    def _train(self, batch_states, batch_probs):
        with self.graph.as_default():
            batch_size = len(batch_states)
            logging.info('Gradient descent with batch size {}'.format(batch_size))
            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(batch_states),
                tf.data.Dataset.from_tensor_slices(batch_probs)))
            dataset = dataset.shuffle(2**16).repeat().batch(self.MINI_BATCH_SIZE)
            iterator = dataset.make_one_shot_iterator()
            next_batch = iterator.get_next()
            validation_inds = random.sample(range(batch_size), min(batch_size, 100))
            validation_feed_dict = {
                self.features: [batch_states[i] for i in validation_inds],
                self.chosen_action: [batch_probs[i] for i in validation_inds]}
            logging.info('Performing {} iterations with mini-batch size {}'.format(self.N_TRAIN_STEPS, self.MINI_BATCH_SIZE))
            loss = []
            for i in range(self.N_TRAIN_STEPS):
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
