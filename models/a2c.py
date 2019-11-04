import gym
import keras.backend as K
import numpy as np
import os
import time

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam


class A2C:
    def __init__(self,
                 env,
                 lr=0.0001,
                 gamma=0.99,
                 state_size=34,
                 cont_acts=6,
                 disc_acts=12,
                 entropy_reg=0.01,
                 network_width=128,
                 model_path='weights/'):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.state_size = state_size
        self.cont_act_dim = cont_acts
        self.disc_act_dim = disc_acts
        self.optimizer = Adam(lr=lr)

        self.cont_actor, self.disc_actor, self.critic = self._build_model(network_width)
        self.cont_opt = self._continuous_optimizer(entropy_reg)
        self.disc_opt = self._discrete_optimizer(entropy_reg)
        self.val_opt = self._critic_optimizer()

        self.model_path = model_path

    def _build_model(self, width):
        state = Input(batch_shape=(None, self.state_size))
        layer2 = Dense(width // 2, input_dim=self.state_size, activation='relu')(state)
        layer3 = Dense(width, activation='relu')(layer2)

        actor = Dense(width, activation='relu')(layer3)
        mu_0 = Dense(self.cont_act_dim, activation='tanh')(actor)
        var_0 = Dense(self.cont_act_dim, activation='softplus')(actor)
        mu = Lambda(lambda x: x * 2)(mu_0)
        var = Lambda(lambda x: x + 0.0001)(var_0)
        disc = Dense(self.disc_act_dim, activation='sigmoid')(actor)

        critic = Dense(width, activation='relu')(layer3)
        val = Dense(1, activation='linear')(critic)

        continuous_head = Model(inputs=state, outputs=[mu, var])
        discrete_head = Model(inputs=state, outputs=disc)
        value_head = Model(inputs=state, outputs=val)

        return continuous_head, discrete_head, value_head

    def _continuous_optimizer(self, reg):
        action = K.placeholder(shape=(None, self.cont_act_dim))
        advantages = K.placeholder(shape=(None,))
        mu, var = self.cont_actor.output

        pdf = 1.0 / K.sqrt(2.0 * np.pi * var) * K.exp(-K.square(action - mu) / (2.0 * var))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2.0 * np.pi * var) + 1.0))
        exp = log_pdf * advantages
        exp = K.sum(exp + reg * entropy)

        loss = -exp
        updates = self.optimizer.get_updates(
                self.cont_actor.trainable_weights, [], loss)

        return K.function(
                [self.cont_actor.input, action, advantages],
                [self.cont_actor.output], updates=updates)

    def _discrete_optimizer(self, reg):
        action = K.placeholder(shape=(None, self.disc_act_dim))
        advantages = K.placeholder(shape=(None,))
        disc = self.disc_actor.output

        weighted_actions = K.sum(action * disc, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages)
        entropy = K.sum(disc * K.log(disc + 1e-10), axis=1)

        loss = reg * entropy - K.sum(eligibility)
        updates = self.optimizer.get_updates(
                self.disc_actor.trainable_weights, [], loss)

        return K.function(
                [self.disc_actor.input, action, advantages],
                [self.disc_actor.output], updates=updates)

    def _critic_optimizer(self):
        discounted_r = K.placeholder(shape=(None,))
        val = self.critic.output

        loss = K.mean(K.square(discounted_r - val))
        updates = self.optimizer.get_updates(
                self.critic.trainable_weights, [], loss)

        return K.function(
                [self.critic.input, discounted_r],
                [self.critic.output], updates=updates)

    def act(self, state):
        state = np.reshape(state, (1, -1))
        mu, var = self.cont_actor.predict(state)
        disc = self.disc_actor.predict(state)

        epsilon = np.random.randn(self.cont_act_dim)
        cont_act = mu + np.sqrt(var) * epsilon
        cont_act = np.clip(cont_act, 0, 1.0)
        disc_act = np.random.random(self.disc_act_dim) < disc

        action = np.concatenate([cont_act, disc_act], axis=1)
        return np.reshape(action, -1)

    def train(self, state, action, reward, next_state, done):
        state = np.reshape(state, (1, -1))
        next_state = np.reshape(state, (1, -1))
        cont_act = action[:self.cont_act_dim]
        disc_act = action[self.cont_act_dim:]

        val = self.critic.predict(state)
        target = reward

        if not done:
            next_val = self.critic.predict(next_state)
            target += self.gamma * next_val
        
        advantage = (target - val)[0][0]
        self.cont_opt([state, cont_act, advantage])
        self.disc_opt([state, disc_act, advantage])
        self.val_opt([state, target])

    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        timestamp = time.asctime().replace(' ', '_')
        self.cont_actor.save_weights(
                os.path.join(self.model_path, timestamp + '_cont_actor.h5'))
        self.disc_actor.save_weights(
                os.path.join(self.model_path, timestamp + '_disc_actor.h5'))
        self.critic.save_weights(
                os.path.join(self.model_path, timestamp + '_critic.h5'))

    def load_model(self, path):
        self.cont_actor.load_weights(path + '_cont_actor.h5')
        self.disc_actor.load_weights(path + '_disc_actor.h5')
        self.critic.load_weights(path + '_critic.h5')
