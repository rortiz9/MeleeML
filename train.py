import argparse

from models.a2c import A2C
from envs.melee_env import MeleeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=False)
    parser.add_argument('--render', default=False, help='Display Dolphin GUI')
    parser.add_argument('--self_play', default=False)
    parser.add_argument('--iso_path', default='../smash.iso', help='Path to MELEE iso')
    parser.add_argument('--model_path', default='weights/', help='Path to store weights')
    parser.add_argument('--load_model', default=None, help='Load model from file')
    parser.add_argument('--num_episdoes', default=10, help='# of games to play')
    args = parser.parse_args()

    env = MeleeEnv(log=args.log,
                   render=args.render,
                   self_play=args.self_play,
                   iso_path=args.iso_path)
    agent = A2C(env, model_path=args.model_path)

    if args.load_model:
        agent.load_model(args.load_model)

    for e in range(args.num_episodes):
        done = False
        state = env.reset()

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state

        agent.save_model()

    env.close()


if __name__ == '__main__':
    main()
