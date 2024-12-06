def viterbi(observations, states, start_probs, transition_probs, emission_probs):
    """
    Perform Viterbi decoding on an HMM.

    Parameters:
    - observations: List of observed symbols.
    - states: List of states in the HMM.
    - start_probs: Dictionary of start probabilities for each state.
    - transition_probs: Dictionary of dictionaries with state transition probabilities.
    - emission_probs: Dictionary of dictionaries with emission probabilities.

    Returns:
    A tuple (best_path, best_path_prob) where:
        - best_path is the most likely sequence of hidden states.
        - best_path_prob is the probability of that sequence.
    """

    T = len(observations)

    # Probability matrix: dp[t][state] = max probability of being in `state` at time t
    dp = [{} for _ in range(T)]
    # Backpointer matrix: to reconstruct the path
    backpointer = [{} for _ in range(T)]

    for s in states:
        dp[0][s] = start_probs[s] * emission_probs[s].get(observations[0], 0)
        backpointer[0][s] = None

    for t in range(1, T):
        for s in states:
            max_prob = 0
            best_prev_state = None
            for prev_s in states:
                prob = dp[t - 1][prev_s] * transition_probs[prev_s][s] * emission_probs[s].get(observations[t], 0)
                if prob > max_prob:
                    max_prob = prob
                    best_prev_state = prev_s
            dp[t][s] = max_prob
            backpointer[t][s] = best_prev_state

    best_path_prob = 0
    best_final_state = None
    for s in states:
        if dp[T - 1][s] > best_path_prob:
            best_path_prob = dp[T - 1][s]
            best_final_state = s

    best_path = [best_final_state]
    for t in range(T - 1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])

    return best_path, best_path_prob


if __name__ == "__main__":
    try:
        num_states = int(input("Enter the number of states: "))
        states = []
        for _ in range(num_states):
            state_name = input("Enter state name: ").strip()
            states.append(state_name)

        num_observations_symbols = int(input("Enter the number of possible observation symbols: "))
        observation_symbols = []
        for _ in range(num_observations_symbols):
            symbol = input("Enter observation symbol: ").strip()
            observation_symbols.append(symbol)

        start_probs = {}
        print("Enter start probabilities for each state (they should sum to 1):")
        for s in states:
            p = float(input(f"P(start in {s}): "))
            start_probs[s] = p

        transition_probs = {}
        print("Enter transition probabilities A(i,j): Probability of going from state i to state j.")
        for s in states:
            transition_probs[s] = {}
            for s2 in states:
                p = float(input(f"P({s}->{s2}): "))
                transition_probs[s][s2] = p

        emission_probs = {}
        print("Enter emission probabilities B(i,o): Probability of emitting observation o in state i.")
        for s in states:
            emission_probs[s] = {}
            for obs in observation_symbols:
                p = float(input(f"P(observation={obs}|state={s}): "))
                emission_probs[s][obs] = p

        print("Now enter the observation sequence you want to decode.")
        seq_len = int(input("Length of observation sequence: "))
        observations = []
        for i in range(seq_len):
            obs = input(f"Observation {i+1}: ").strip()
            if obs not in observation_symbols:
                print(f"Warning: {obs} is not in the known observation symbols. Its probability defaults to 0 if not defined.")
            observations.append(obs)

        best_path, best_path_prob = viterbi(observations, states, start_probs, transition_probs, emission_probs)

        print(f"Best hidden state sequence: {best_path}")
        print(f"Probability of the best path: {best_path_prob}")
    except Exception as e:
        print(e)
