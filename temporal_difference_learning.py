import logging

import numpy as np
from numpy.typing import NDArray

from secret_envs_wrapper import SecretEnv0, SecretEnv1

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Sarsa:
    def __init__(
        self,
        env: SecretEnv0 | SecretEnv1,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        num_episodes: int = 1000,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.Q = np.zeros((env.num_states(), env.num_actions()))
        self.best_policy: NDArray[np.int_] = np.array([], dtype=int)
        self.best_score = float("-inf")
        logger.info(
            f"Initialized SarsaAgent with alpha={alpha}, gamma={gamma}, epsilon={epsilon}, num_episodes={num_episodes}"
        )

    def epsilon_greedy_policy(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            available_actions = self.env.available_actions()
            action = np.random.choice(available_actions)
            logger.debug(
                f"Epsilon-greedy: Chose random action {action} for state {state}"
            )
            return action
        else:
            available_actions = self.env.available_actions()
            q_values = self.Q[state, available_actions]
            action = available_actions[np.argmax(q_values)]
            logger.debug(
                f"Epsilon-greedy: Chose greedy action {action} for state {state}"
            )
            return action

    def train(self) -> NDArray[np.int_]:
        logger.info("Starting training")
        for episode in range(self.num_episodes):
            self.env.reset()
            state = self.env.state_id()
            action = self.epsilon_greedy_policy(state)
            logger.debug(
                f"Episode {episode + 1}: Starting state {state}, action {action}"
            )

            done = False
            while not done:
                self.env.step(action)
                next_state = self.env.state_id()
                reward = self.env.score()
                done = self.env.is_game_over()
                next_action = self.epsilon_greedy_policy(next_state)

                # Update Q-value
                old_q = self.Q[state, action]
                self.Q[state, action] += self.alpha * (
                    reward
                    + self.gamma * self.Q[next_state, next_action]
                    - self.Q[state, action]
                )
                logger.debug(
                    f"Updated Q[{state}, {action}] from {old_q} to {self.Q[state, action]}"
                )

                state = next_state
                action = next_action

            episode_score = self.env.score()
            logger.info(
                f"Episode {episode + 1} completed with final score: {episode_score}, Best score: {self.best_score}"
            )

            # Update best policy if current episode score is higher
            if episode_score > self.best_score:
                self.best_score = episode_score
                self.best_policy = self.get_policy()
                logger.info(f"New best policy found with score: {self.best_score}")

        logger.info("Training completed")
        return self.best_policy

    def get_policy(self) -> NDArray[np.int_]:
        logger.info("Generating policy from Q-values")
        policy = np.zeros(self.env.num_states(), dtype=int)
        for state in range(self.env.num_states()):
            available_actions = self.env.available_actions()
            q_values = self.Q[state, available_actions]
            policy[state] = available_actions[np.argmax(q_values)]
            logger.debug(f"Policy for state {state}: {policy[state]}")
        return policy

    def get_best_policy(self) -> NDArray[np.int_]:
        logger.info("Returning best policy")
        return self.best_policy if self.best_policy.size > 0 else self.get_policy()

    def get_Q(self) -> NDArray[np.float64]:
        logger.info("Returning Q-values")
        return self.Q
