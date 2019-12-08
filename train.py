import argparse
import melee
import numpy as np
import time

from dataset import get_data_from_logs
from models.a2c import A2C
from envs.melee_env import MeleeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=time.asctime(), help='Name of this run')
    parser.add_argument('--log', default=False, action='store_true')
    parser.add_argument(
            '--render', default=False, action='store_true', help='Display Dolphin GUI')
    parser.add_argument(
            '--eval', default=False, action='store_true', help='Run evaluation games')
    parser.add_argument(
            '--warm_start', default=None, help='Directory of human replay data')
    parser.add_argument('--self_play', default=False, action='store_true')
    parser.add_argument(
            '--iso_path', default='../smash.iso', help='Path to MELEE iso')
    parser.add_argument(
            '--model_path', default='weights/', help='Path to store weights')
    parser.add_argument('--load_model', default=None, help='Load model from file')
    parser.add_argument(
            '--num_episodes', type=int, default=10, help='# of games to play')
    args = parser.parse_args()

    name = args.name.replace(' ', '_')
    env = MeleeEnv(log=args.log,
                   render=args.render,
                   self_play=args.self_play,
                   iso_path=args.iso_path)
    agent = A2C(env, model_path=args.model_path)

    if args.load_model:
        agent.load_model(args.load_model)

    if args.warm_start:
        # TODO: converting one-hot actions to joystick inputs
        states, actions = get_data_from_logs(args.warm_start)

        for i in range(states.shape[0]):
            state = states[i]
            action = np.zeros(16)
            action[:6] = actions[i][10:] / 255
            action[6:11] = actions[i][:5]
            action[11:15] = actions[i][6:10]

            if not np.any(action[6:15]):
                action[15] = 1.0

            if state[5] == 0 or state[20] == 0:
                continue

            next_state = states[i + 1]
            done = False
            p1_score = (1000 * (next_state[5] - state[5]) -
                       (next_state[4] - state[4]))
            p2_score = (1000 * (next_state[20] - state[20]) -
                       (next_state[19] - state[19]))

            if next_state[5] == 0 or next_state[20] == 0:
                done = True

            reward = p1_score - p2_score
            agent.train(state, action, reward, next_state, done)

            if args.eval and i % 36000 == 0:
                eval_score = 0
                eval_done = False
                eval_state = env.reset()

                while not eval_done:
                    eval_action = agent.act(eval_state)
                    eval_state, eval_reward, eval_done = env.step(eval_action)
                    eval_score += eval_reward

                print(eval_score)

                while (env.gamestate.menu_state in [
                    melee.enums.Menu.IN_GAME, melee.enums.Menu.SUDDEN_DEATH] and (
                        env.gamestate.ai_state.stock == 0 or
                        env.gamestate.opponent_state.stock == 0)):
                    env.gamestate.step()

    for e in range(args.num_episodes):
        done = False
        states = []
        actions = []
        rewards = []
        state = env.reset()

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        for i in range(len(states) - 1):
            agent.train(states[i], actions[i], rewards[i], states[i + 1], False)

        agent.train(states[-1], actions[-1], rewards[-1], states[-1], True)
        agent.save_model(name, e)

        while (env.gamestate.menu_state in [
            melee.enums.Menu.IN_GAME, melee.enums.Menu.SUDDEN_DEATH] and (
                env.gamestate.ai_state.stock == 0 or
                env.gamestate.opponent_state.stock == 0)):
            env.gamestate.step()

    env.close()


if __name__ == '__main__':
    main()
