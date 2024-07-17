import pickle
from planning import DynaQ, QLearning
from secret_envs_wrapper import SecretEnv0
from temporal_difference_learning import Sarsa


if __name__ == "__main__":
    gamma: float = 0.999  # monte_carlo / dynamic_programming
    nb_iter: int = 10000  # monte_carlo
    max_steps: int = 10  # monte_carlo
    epsilon = 0.1  # monte_carlo
    theta = 1e-6  # dynamic_programming
    max_iterations = 1000  # dynamic_programming

    # env = LineWorld    # best score 1 : gamma 0.999 iter 10 steps 10
    # env = GridWorld
    env = SecretEnv0()
    # env = SecretEnv0() # best score 7 : gamma 0.999 iter 100000 steps 100
    # env = SecretEnv1() # best score 8 : gamma 0.999 iter 100000 steps 100

    # monte_carlo.naive_monte_carlo_with_exploring_starts(env, gamma, nb_iter, max_steps)
    # monte_carlo.on_policy_first_visit_mc_control(env, gamma, nb_iter, max_steps, epsilon)
    # monte_carlo.off_policy_mc_control(env, gamma, nb_iter, max_steps, epsilon)

    # Test Policy Iteration
    # policy_iteration = PolicyIteration(env, gamma, theta, max_iterations)
    # optimal_policy_pi = policy_iteration.run()
    # print("Optimal Policy from Policy Iteration:")
    # print(optimal_policy_pi)

    # Test Value Iteration
    # value_iteration = ValueIteration(env, gamma, theta, max_iterations)
    # optimal_policy_vi = value_iteration.run()
    # print("Optimal Policy from Value Iteration:")
    # print(optimal_policy_vi)

    # Test SARSA
    # alpha = 0.1
    # num_episodes = 1000
    # sarsa_agent = SarsaAgent(
    #     env, alpha=alpha, gamma=gamma, epsilon=epsilon, num_episodes=num_episodes
    # )
    # sarsa_agent.train()

    # print("Table Q apprise par SARSA:")
    # print(sarsa_agent.get_Q())

    # print("Politique apprise par SARSA:")
    # print(sarsa_agent.get_policy())

    # alpha = 0.1
    # num_episodes = 1000
    # q_learning = QLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon)

    # q_learning.train(num_episodes)

    # print("Table Q apprise par Q-Learning:")
    # print(q_learning.get_q_table())

    # # Pour obtenir la politique apprise par Q-Learning
    # def get_policy(agent: QLearning) -> np.ndarray:
    #     return np.argmax(agent.get_q_table(), axis=1)

    # print("Politique apprise par Q-Learning:")
    # print(get_policy(q_learning))

    # dynaQ
    # Test DynaQ

env = SecretEnv0()
agent = DynaQ(env, alpha=0.1, gamma=0.95, epsilon=0.1, dyna_planning_steps=10)

initial_state = env.state_id()
num_episodes = 1000
best_policy = agent.train(num_episodes)
print("Best policy:")
with open("./pckl/monty_hall_v1/dyna_q.pkl", "wb") as f:
    pickle.dump(best_policy, f)
