def _strip_state(state):
    state = np.array(state)
    print("in: ", state)
    state = state[~np.isnan(state)]
    print("out: ", state)
    return state

