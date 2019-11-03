import gym
import keras.backend as K
import numpy as np
import os
import tensorflow as tf
import time

from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical


class A2C:
    def __init__(self,
                 env,
                 lr=0.0001,
                 gamma=0.99,
                 epsilon=0.1,
                 rho=0.99,
                 entropy_reg=0.01,
                 network_width=128
                 model_path='weights/'):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.rms_optimizer = RMSprop(lr=lr, epsilon=epsilon, rho=rho)

        self.shared = self._shared_layers(network_width)
        self.actor_head = self._actor(network_width)
        self.critic_head = self._critic(network_width)

        self.actor_opt = self._actor_optimizer(entropy_reg)
        self.critic_opt = self._critic_optimizer()

        self.model_path = model_path

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

    def act(self, state):
        return np.random.choice(np.arange(self.env.action_space.shape[0]),
                                1,
                                p=self.actor_head.predict(state).ravel())[0]

    def _discount(self, rewards):
        cumul_r = 0
        discounted_r = np.zeros_like(rewards)

        for t in range(len(rewards) - 1, -1, -1):
            cumul_r = rewards[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r

        return discounted_r

    def _train_models(self, states, actions, rewards, done):
        discounted_rewards = self._discount(rewards)
        state_values = self.critic_head.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        self.actor_opt([states, actions, advantages])
        self.critic_opt([states, discounted_rewards])

    def train(self, episodes, summary_writer):
        for e in range(episodes):
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []

            while not done:
                action = self.act(old_state)
                new_state, reward, done, _ = env.step(action)
                actions.append(to_categorical(action, self.env.action_space.shape[0]))
                rewards.append(reward)
                states.append(old_state)
                old_state = new_state
                cumul_reward += reward
                time += 1
            
            self._train_models(states, actions, rewards, done)
            self.save_models()

            score = tf.Summary(value=[
                tf.Summary.Value(tag='score', simple_value=cumul_reward)])
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

    def save_models(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        timestamp = time.asctime().replace(' ', '_')
        self.actor_head.save_weights(
                os.path.join(self.model_path, timestamp + '_actor.h5'))
        self.critic_head.save_weights(
                os.path.join(self.model_path, timestamp + '_critic.h5'))

    def load_models(self, actor_path, critic_path):
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            return

        self.actor_head.load_weights(actor_path)
        self.critic_head.load_weights(critic_path)
