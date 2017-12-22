import tensorflow as tf
import game
import random
import learning
import logging
from util import Progress


class FullyConnectedEvaluator(learning.LearningEvaluator):
    N_FEATURES = 4
    INPUT_SIZE = game.ROWS * game.COLUMNS * N_FEATURES

    COUNTER = -1
    @classmethod
    def _increment(cls):
        cls.COUNTER += 1
        return cls.COUNTER

    def _matrix_variable(self, input_dim, output_dim, parameters, name):
        if parameters is not None:
            initial = tf.Variable(parameters[name])
        else:
            stddev = (2 / (input_dim + output_dim)) ** 0.5
            initial = tf.truncated_normal([input_dim, output_dim], stddev=stddev)
        self.variables[name] = tf.Variable(initial)
        return self.variables[name]

    def _affine_variable(self, dim, parameters, name):
        if parameters is not None:
            initial = parameters[name]
        else:
            initial = tf.zeros([dim])
        self.variables[name] = tf.Variable(initial)
        return self.variables[name]

    def __init__(self, n_layers, n_layer_neurons, parameters=None, regularization=1e-3, mini_batch_size=2048, n_train_steps=1000):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._n_layers = n_layers
            self._n_layer_neurons = n_layer_neurons
            self._regularization = regularization
            self._mini_batch_size = mini_batch_size
            self._n_train_steps = n_train_steps
            self.index = self._increment()
            self.features = tf.placeholder(tf.float32, [None, self.N_FEATURES, game.ROWS * game.COLUMNS])
            self.chosen_action = tf.placeholder(tf.float32, [None, game.COLUMNS])
            self.value = tf.placeholder(tf.float32, [None, 1])
            self.variables = {}
            last_layer = tf.reshape(self.features, [-1, self.INPUT_SIZE])
            for i in range(n_layers):
                matrix = self._matrix_variable(
                    self.INPUT_SIZE if i == 0 else n_layer_neurons,
                    n_layer_neurons,
                    parameters, 'matrix_{}'.format(i)
                )
                affine = self._affine_variable(
                    n_layer_neurons,
                    parameters, 'affine_{}'.format(i)
                )
                last_layer = tf.nn.relu(tf.matmul(last_layer, matrix) + affine)
            policy_matrix = self._matrix_variable(
                n_layer_neurons, game.COLUMNS,
                parameters, 'policy_matrix')
            policy_affine = self._affine_variable(
                game.COLUMNS,
                parameters, 'policy_affine'
            )
            action_logits = tf.add(tf.matmul(last_layer, policy_matrix), policy_affine)
            self.action_probs = tf.nn.softmax(action_logits)
            self.policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.chosen_action, logits=action_logits))
            value_matrix = self._matrix_variable(
                n_layer_neurons, 1,
                parameters, 'value_matrix')
            value_affine = self._affine_variable(
                1,
                parameters, 'value_affine'
            )
            self.value_pred = tf.nn.tanh(tf.add(tf.matmul(last_layer, value_matrix), value_affine))
            self.mse_loss = tf.reduce_mean(tf.squared_difference(self.value_pred, self.value))
            self.loss = self.policy_loss + self.mse_loss + \
                regularization * sum(tf.nn.l2_loss(var) for var in self.variables.values())
            opt = tf.train.MomentumOptimizer(learning_rate=0.05, momentum=0.9, use_nesterov=True)
            self.train_op = opt.minimize(self.loss)
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

    def evaluate(self, features):
        with self.graph.as_default():
            if random.random() < 0.5:
                features = game.reflect_columns(features)
            action_probs, value_pred = self.session.run([self.action_probs, self.value_pred], feed_dict={
                self.features: [features]
            })
            return action_probs[0], value_pred[0][0]

    def train(self, batch_states, batch_probs, batch_values):
        with self.graph.as_default():
            parameters = {name: self.session.run(var)
                          for name, var in self.variables.items()}
        obj = FullyConnectedEvaluator(n_layers=self._n_layers,
                                      n_layer_neurons=self._n_layer_neurons,
                                      parameters=parameters,
                                      regularization=self._regularization,
                                      mini_batch_size=self._mini_batch_size,
                                      n_train_steps=self._n_train_steps)
        obj._train(batch_states, batch_probs, batch_values)
        return obj

    def get_name(self):
        return 'full_{}'.format(self.index)

    def _train(self, batch_states, batch_probs, batch_values):
        with self.graph.as_default():
            batch_size = len(batch_states)
            logging.info('Gradient descent with batch size {}'.format(batch_size))
            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(batch_states),
                tf.data.Dataset.from_tensor_slices(batch_probs),
                tf.data.Dataset.from_tensor_slices(batch_values)))
            dataset = dataset.shuffle(2**16).repeat().batch(self._mini_batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_batch = iterator.get_next()
            validation_inds = random.sample(range(batch_size), min(batch_size, 100))
            validation_feed_dict = {
                self.features: [batch_states[i] for i in validation_inds],
                self.chosen_action: [batch_probs[i] for i in validation_inds],
                self.value: [batch_values[i] for i in validation_inds]}
            logging.info('Performing {} iterations with mini-batch size {}'.format(self._n_train_steps, self._mini_batch_size))
            loss = []
            for i in Progress(range(self._n_train_steps)):
                next_states, next_probs, next_values = self.session.run(next_batch)
                feed_dict = {
                    self.features: next_states,
                    self.chosen_action: next_probs,
                    self.value: next_values
                }
                if i % 100 == 0:
                    loss.append(self.session.run(self.loss, feed_dict=validation_feed_dict))
                self.session.run(self.train_op, feed_dict=feed_dict)
            loss.append(self.session.run(self.loss, feed_dict=validation_feed_dict))
            logging.info('Training losses: {}'.format(', '.join('{:.3f}'.format(x) for x in loss)))
