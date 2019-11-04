from models.a2c import A2C
from envs.melee_env import MeleeEnv


def main():
    env = MeleeEnv(render=True)
    agent = A2C(env)

    for e in range(10):
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
