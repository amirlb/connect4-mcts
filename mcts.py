import game
import numpy as np


class Evaluator(object):
    def get_name(self):
        "Return name of the object"

    def evaluate(self, features):
        "Return probabilities vector and estimated value"
        raise NotImplementedError()


class Uninformative(Evaluator):
    def get_name(self):
        return "naive"

    def evaluate(self, features):
        return [1 / game.COLUMNS] * game.COLUMNS, 0


class MovesGraph(object):

    class EdgeData(object):
        def __init__(self, player, next_state, prob, value):
            self.value_sign = {'A': 1, 'B': -1}[player]
            self.next_state = next_state
            self.prior_prob = prob
            self.n_visits = 0
            self.total_value = 0
            self.value = value * self.value_sign
            self.player_value = self.value

        def add_noise(self, epsilon, noise):
            self.prior_prob = (1 - epsilon) * self.prior_prob + epsilon * noise

        def update(self, value):
            self.n_visits += 1
            self.total_value += value
            self.value = self.total_value / self.n_visits
            self.player_value = self.value * self.value_sign

    def __init__(self, evaluator, epsilon=0.25, puct_const=2.8):
        self._evaluator = evaluator
        self._cache = {}  # dict from state to node, where node is dict from action to EdgeData
        self._epsilon = epsilon
        self._puct_const = 0.85

    def reset(self):
        self._cache = {}

    def choose_action(self, state, features, n_playouts):
        if state not in self._cache:
            self._create_node(state, features)
        node = self._cache[state]
        self._add_noise(node)
        for i in range(n_playouts):
            self._expand(node)
        most_visits = max(edge.n_visits for edge in node.values())
        best_actions = [action for action, edge in node.items() if edge.n_visits == most_visits]
        probs = np.zeros(game.COLUMNS)
        probs[best_actions] = 1 / len(best_actions)
        return probs.tolist(), np.random.choice(best_actions)

    def _create_node(self, state, features):
        _, player = state
        probs, value = self._evaluator.evaluate(features)
        factor = 1 / sum(probs[action] for action in game.actions(state).keys())
        self._cache[state] = {
            action: self.EdgeData(player, next_state, probs[action] * factor, value)
            for action, next_state in game.actions(state).items()
        }
        return value

    def _add_noise(self, node):
        noise = np.random.dirichlet(np.ones(game.COLUMNS) / game.COLUMNS)
        factor = 1 / sum(noise[action] for action in node.keys())
        for action, edge in node.items():
            edge.add_noise(self._epsilon, noise[action] * factor)

    def _expand(self, node):
        edge = self._choose_edge(node)
        if edge.next_state in self._cache:
            value = self._expand(self._cache[edge.next_state])
        elif edge.next_state in game.OUTCOMES:
            value = game.OUTCOMES[edge.next_state]
        else:
            features = game.encode_board(edge.next_state)
            value = self._create_node(edge.next_state, features)
        edge.update(value)
        return value

    def _choose_edge(self, node):
        node_visits = sum(edge.n_visits for edge in node.values())
        scores = {
            action: edge.player_value + self._puct_const * edge.prior_prob * node_visits**0.5 / (1 + edge.n_visits)
            for action, edge in node.items()
        }
        best_score = max(scores.values())
        best_actions = [action for action, score in scores.items() if score == best_score]
        action = np.random.choice(best_actions)
        return node[action]


class MCTS_Player(game.Player):
    def __init__(self, evaluator, n_playouts, **kwargs):
        self.evaluator_name = evaluator.get_name()
        self._moves_graph = MovesGraph(evaluator, **kwargs)
        self._n_playouts = n_playouts

    def reset(self):
        self._moves_graph.reset()

    def choose_action(self, match):
        state = match.states[-1]
        features = match.encoded_boards[-1]
        return self._moves_graph.choose_action(state, features, self._n_playouts)
