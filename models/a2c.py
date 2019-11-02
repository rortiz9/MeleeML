from collections import deque
import gym
import keras.backend as K
import numpy as np

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import RMSprop


class A2C:
    def __init__(self,
                 env,
                 session,
                 lr=0.0001,
                 gamma=0.99,
                 epsilon=0.1,
                 rho=0.99,
                 entropy_reg=0.01,
                 network_width=128):
        self.env = env
        self.session = session
        self.lr = lr
        self.gamma = gamma
        self.rms_optimizer = RMSprop(lr=lr, epsilon=epsilon, rho=rho)

        self.shared = self._shared_layers(network_width)
        self.actor_head = self._actor(network_width)
        self.critic_head = self._critic(network_width)

        self.actor_opt = self._actor_optimizer(entropy_reg)
        self.critic_opt = self._critic_optimizer()

    def _shared_layers(self, width):
        state = Input(shape=self.env.observation_space.shape)
        layer1 = Flatten()(state)
        layer2 = Dense(width / 2, activation='relu')(layer1)
        layer3 = Dense(width, activation='relu')(layer2)
        return Model(state, layer3)

    def _actor(self, width):
        layer4 = Dense(width, activation='relu')(self.shared.output)
        output = Dense(self.env.action_space.shape[0], activation='softmax')(layer4)
        return Model(self.shared.input, output)

    def _critic(self, width):
        layer4 = Dense(width, activation='relu')(self.shared.output)
        output = Dense(1, activation='linear')(layer4)
        return Model(self.shared.input, output)

    def _actor_optimizer(self, reg):
        action = K.placeholder(shape=(None, self.env.action_space.shape[0]))
        advantages = K.placeholder(shape=(None,))

        weighted_actions = K.sum(action * self.actor_head.output, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages)
        entropy = K.sum(self.actor_head.output
                * K.log(self.actor_head.output + 1e-10), axis=1)

        loss = reg * entropy - K.sum(eligibility)
        updates = self.rms_optimizer.get_updates(
                self.actor_head.trainable_weights, [], loss)

        return K.function(
                [self.actor_head.input, action, advantages], [], updates=updates)

    def _critic_optimizer(self):
        discounted_r = K.placeholder(shape=(None,))

        loss = K.mean(K.square(discounted_r - self.critic_head.output))
        updates = self.rms_optimizer.get_updates(
                self.critic_head.trainable_weights, [], loss)

        return K.function(
                [self.critic_head.input, self.discounted_r], [], updates=updates)
