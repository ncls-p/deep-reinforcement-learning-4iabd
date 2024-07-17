import pickle
from env import MontyHallLvl2
import monte_carlo


# Example usage
env_type = MontyHallLvl2  # Replace with your environment class
gamma = 0.999
nb_iter = 10000
max_steps = 10
epsilon = 0.1

""" Pi, V, Q = on_policy_first_visit_mc_control(env_type, gamma, nb_iter, max_steps, epsilon) """
Pi = monte_carlo.on_policy_first_visit_mc_control(
    env_type, gamma, nb_iter, max_steps, epsilon
)

# Save the policy, value function, and action-value function
with open("./pckl/policy.pkl", "wb") as f:
    pickle.dump(Pi, f)

""" with open('./pckl/value_function.pkl', 'wb') as f:
    pickle.dump(V, f) """

""" with open('/pckl/action_value_function.pkl', 'wb') as f:
    pickle.dump(Q, f) """

# Load the policy, value function, and action-value function
with open("./pckl/policy.pkl", "rb") as f:
    Pi_loaded = pickle.load(f)

""" with open('./pckl/value_function.pkl', 'rb') as f:
    V_loaded = pickle.load(f) """

""" with open('./pckl/action_value_function.pkl', 'rb') as f:
    Q_loaded = pickle.load(f) """

# Verify loaded data
print(Pi_loaded == Pi)  # Should print True
""" print(V_loaded == V)    # Should print True """
""" print(Q_loaded == Q)    # Should print True """


# Execute the loaded policy in the environment
def run_policy(env_type, Pi, max_steps):
    env = env_type()  # Initialize the environment
    env.reset()
    env.display()
    steps_count = 0

    while not env.is_game_over() and steps_count < max_steps:
        s = env.state_id()
        if s in Pi:
            a = Pi[s]
            env.step(a)
            env.display()
            steps_count += 1
        else:
            print("State not found in policy:", s)
            break

    print("Final Score:", env.score())


# Run the loaded policy in the environment
run_policy(env_type, Pi_loaded, max_steps)
