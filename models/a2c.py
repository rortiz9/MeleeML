import gym
import keras.backend as K
import numpy as np
import os
import tensorflow as tf
import time

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical


class A2C:
    def __init__(self,
                 env,
                 lr=0.0001,
                 gamma=0.99,
                 cont_acts=6,
                 disc_acts=12,
                 entropy_reg=0.01,
                 network_width=128,
                 model_path='weights/'):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.state_shape = env.observation_space.shape
        self.cont_act_dim = cont_acts
        self.disc_act_dim = disc_acts
        self.optimizer = Adam(lr=lr)

        self.model = self._build_model(network_width)
        self.cont_opt = self._continuous_optimizer(entropy_reg)
        self.disc_opt = self._discrete_optimizer(entropy_reg)
        self.val_opt = self._critic_optimizer()

        self.model_path = model_path

    def _build_model(self, width):
        state = Input((1,) + self.state_shape)
        layer2 = Dense(width // 2, activation='relu')(state)
        layer3 = Dense(width, activation='relu')(layer2)

        actor = Dense(width, activation='relu')(layer3)
        mu = Dense(self.cont_act_dim, activation='tanh')(actor)
        var = Dense(self.cont_act_dim, activation='softplus')(actor)
        disc = Dense(self.disc_act_dim, activation='sigmoid')(actor)

        critic = Dense(width, activation='relu')(layer3)
        val = Dense(1, activation='linear')(critic)

        return Model(inputs=state, outputs=[mu, var, disc, val])

    def _continuous_optimizer(self, reg):
        action = K.placeholder(shape=(None, self.cont_act_dim))
        advantages = K.placeholder(shape=(None,))
        mu, var, _, _ = self.model.output

        pdf = 1.0 / K.sqrt(2.0 * np.pi * var) * K.exp(-K.square(action - mu) / (2.0 * var))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2.0 * np.pi * var) + 1.0))
        exp = log_pdf * advantages
        exp = K.sum(exp + reg * entropy)

        loss = -exp
        updates = self.optimizer.get_updates(self.model.trainable_weights, [], loss)

        return K.function(
                [self.model.input, action, advantages], [], updates=updates)

    def _discrete_optimizer(self, reg):
        action = K.placeholder(shape=(None, self.disc_act_dim))
        advantages = K.placeholder(shape=(None,))
        _, _, disc, _ = self.model.output

        weighted_actions = K.sum(action * disc, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages)
        entropy = K.sum(disc * K.log(disc + 1e-10), axis=1)

        loss = reg * entropy - K.sum(eligibility)
        updates = self.optimizer.get_updates(self.model.trainable_weights, [], loss)

        return K.function(
                [self.model.input, action, advantages], [], updates=updates)

    def _critic_optimizer(self):
        discounted_r = K.placeholder(shape=(None,))
        _, _, _, val = self.model.output

        loss = K.mean(K.square(discounted_r - val))
        updates = self.optimizer.get_updates(self.model.trainable_weights, [], loss)

        return K.function(
                [self.model.input, discounted_r], [], updates=updates)

    def act(self, state):
        mu, var, disc, _ = self.model.predict(state)

        epsilon = np.random.randn(self.cont_act_dim)
        cont_act = mu + np.sqrt(var) * epsilon
        cont_act = np.clip(cont_act, 0, 1.0)

        disc_act = np.random.random(self.disc_act_dim) < disc
        return np.concatenate([cont_act, disc_act])

    def train(self, state, action, reward, next_state, done):
        cont_act = action[:self.cont_act_dim]
        disc_act = action[self.cont_act_dim:]

        _, _, _, val = self.model.predict(state)
        target = reward

        if not done:
            _, _, _, next_val = self.model.predict(next_state)
            target += self.gamma * next_val
        
        advantage = target - val
        self.cont_opt([state, cont_act, advantage])
        self.disc_opt([state, disc_act, advantage])
        self.val_opt([state, target])

    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        timestamp = time.asctime().replace(' ', '_')
        self.model.save_weights(
                os.path.join(self.model_path, timestamp + '_model.h5'))

    def load_model(self, path):
        if not os.path.exists(path):
            return

        self.model.load_weights(path)
