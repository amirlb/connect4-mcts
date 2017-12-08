import random
import game


class Player(object):
    def choose_action(self, state):
        raise NotImplementedError()


class RandomPlayer(Player):
    def choose_action(self, state):
        return random.choice(list(game.State.actions(state).keys()))
