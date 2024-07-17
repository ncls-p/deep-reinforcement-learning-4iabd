# Reinforcement Learning Algorithms

This project implements various reinforcement learning algorithms for different environments. The main components are:

## Files

1. `planning.py`: Contains implementations of Q-Learning and Dyna-Q algorithms.
2. `temporal_difference_learning.py`: Implements the SARSA (State-Action-Reward-State-Action) algorithm.
3. `dynamic_programming.py`: Includes implementations of Policy Iteration and Value Iteration algorithms.
4. `monte_carlo.py`: Contains various Monte Carlo methods for reinforcement learning.
5. `env.py`: Defines different environments (LineWorld, GridWorld, and TwoRoundRockPaperScissors) for testing the algorithms.

## Algorithms

### Planning (`planning.py`)

- **Q-Learning**: A model-free reinforcement learning algorithm that learns the optimal action-value function.
- **Dyna-Q**: An integrated planning, acting, and learning algorithm that uses a model to generate simulated experiences.

### Temporal Difference Learning (`temporal_difference_learning.py`)

- **SARSA**: An on-policy TD control algorithm for estimating action-value functions.

### Dynamic Programming (`dynamic_programming.py`)

- **Policy Iteration**: An algorithm that alternates between policy evaluation and policy improvement steps.
- **Value Iteration**: An algorithm that combines policy evaluation and policy improvement in a single step.

### Monte Carlo Methods (`monte_carlo.py`)

- **Naive Monte Carlo with Exploring Starts**: A Monte Carlo method that uses exploring starts for state-action pair visits.
- **On-policy First-Visit Monte Carlo Control**: An on-policy Monte Carlo method for control problems.
- **Off-policy Monte Carlo Control**: An off-policy Monte Carlo method using importance sampling.

## Environments (`env.py`)

- **LineWorld**: A simple 1D environment with 5 states.
- **GridWorld**: A 2D grid environment with rewards and obstacles.
- **TwoRoundRockPaperScissors**: A two-round game of Rock-Paper-Scissors against a random opponent.
- **MontyHallLvl1**: An environement with 3 doors.
- **MontyHallLvl2**: A Environement with 5 doors.
- **SecretEnv0**: **A secret environment**
- **SecretEnv1**: A secret environment.

## Usage

To use these algorithms, import the desired classes from the respective files and create instances with the appropriate environment. For example:

```python
from planning import QLearning
from env import GridWorld

env = GridWorld()
q_learning = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1)
q_learning.train(num_episodes=1000)
```

Refer to the individual files for more detailed usage instructions and parameter descriptions.

## Dependencies

- NumPy
- tqdm

Make sure to install the required dependencies before running the algorithms.
