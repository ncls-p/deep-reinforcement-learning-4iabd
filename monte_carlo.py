from collections import defaultdict

import numpy as np
from tqdm import tqdm


def naive_monte_carlo_with_exploring_starts(env_type, gamma, nb_iter, max_steps):
    Pi = {}
    Q = {}
    Returns = {}

    for it in tqdm(range(nb_iter)):
        env = env_type.from_random_state()

        is_first_action = True
        trajectory = []
        steps_count = 0
        while not env.is_game_over() and steps_count < max_steps:
            s = env.state_id()
            aa = env.available_actions()

            if s not in Pi:
                Pi[s] = np.random.choice(aa)

            if is_first_action:
                a = np.random.choice(aa)
                is_first_action = False
            else:
                a = Pi[s]

            prev_score = env.score()

            env.step(a)
            r = env.score() - prev_score

            trajectory.append((s, a, r, aa))
            steps_count += 1

        G = 0
        for t, (s, a, r, aa) in reversed(list(enumerate(trajectory))):
            G = gamma * G + r

            if all(
                map(lambda triplet: triplet[0] != s or triplet[1] != a, trajectory[:t])
            ):
                if (s, a) not in Returns:
                    Returns[(s, a)] = []
                Returns[(s, a)].append(G)
                Q[(s, a)] = np.mean(Returns[(s, a)])

                best_a = None
                best_a_score = 0.0
                for a in aa:
                    if (s, a) not in Q:
                        Q[(s, a)] = np.random.random()
                    if best_a is None or Q[(s, a)] > best_a_score:
                        best_a = a
                        best_a_score = Q[(s, a)]

                Pi[s] = best_a

    return Pi


def on_policy_first_visit_mc_control(env_type, gamma, nb_iter, max_steps, epsilon):
    Q = defaultdict(lambda: defaultdict(float))
    Returns = defaultdict(list)

    for it in tqdm(range(nb_iter)):
        env = env_type.from_random_state()
        trajectory = []
        G = 0
        steps_count = 0
        visited_state_actions = set()

        while not env.is_game_over() and steps_count < max_steps:
            s = env.state_id()
            aa = env.available_actions()

            if s not in Q:
                Q[s] = {a: 0 for a in aa}

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                a = max(Q[s], key=Q[s].get)

            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            trajectory.append((s, a, r))
            steps_count += 1

        for t in range(len(trajectory)):
            s, a, r = trajectory[t]
            G = gamma * G + r

            if (s, a) not in visited_state_actions:
                visited_state_actions.add((s, a))
                Returns[(s, a)].append(G)
                Q[s][a] = np.mean(Returns[(s, a)])

    # Compute the policy Pi from Q
    Pi = {s: max(Q[s], key=Q[s].get) for s in Q}
    return Pi


def off_policy_mc_control(env_type, gamma, nb_iter, max_steps, epsilon):
    Q = defaultdict(lambda: defaultdict(float))
    C = defaultdict(lambda: defaultdict(float))

    # Behavior policy: epsilon-greedy policy
    def behavior_policy(state):
        if np.random.random() < epsilon:
            return np.random.choice(env.available_actions())
        else:
            return max(
                Q[state],
                key=Q[state].get,
                default=np.random.choice(env.available_actions()),
            )

    # Target policy: greedy policy
    def target_policy(state):
        return max(
            Q[state],
            key=Q[state].get,
            default=np.random.choice(env.available_actions()),
        )

    for it in tqdm(range(nb_iter)):
        env = env_type.from_random_state()
        trajectory = []
        G = 0

        while not env.is_game_over() and len(trajectory) < max_steps:
            s = env.state_id()
            a = behavior_policy(s)
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            trajectory.append((s, a, r))

        W = 1.0
        for s, a, r in reversed(trajectory):
            G = gamma * G + r
            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])
            if a != target_policy(s):
                break
            W *= 1.0 / (
                epsilon / len(env.available_actions()) + (1 - epsilon)
                if a == target_policy(s)
                else 0
            )

    # Compute the policy Pi from Q
    Pi = {s: max(Q[s], key=Q[s].get) for s in Q}
    return Pi
