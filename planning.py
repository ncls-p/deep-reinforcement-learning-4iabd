import logging
import random
from typing import List, Optional, Tuple

import numpy as np
from env import MontyHallLvl1, MontyHallLvl2

from secret_envs_wrapper import SecretEnv0, SecretEnv1

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QLearning:
    def __init__(
        self,
        env: SecretEnv0 | SecretEnv1 | MontyHallLvl1 | MontyHallLvl2,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.env = env
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Paramètre d'exploration
        self.n_states = env.num_states()
        self.n_actions = env.num_actions()
        self.Q = np.zeros((self.n_states, self.n_actions))  # Table Q initialisée à zéro
        logger.info(
            f"Initialized QLearning with alpha={alpha}, gamma={gamma}, epsilon={epsilon}"
        )

    def choose_action(self, state: int) -> int:
        available_actions = self.env.available_actions()
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(available_actions)
            logger.debug(
                f"Chose random action {action} for state {state} (exploration)"
            )
        else:
            q_values = self.Q[state, available_actions]
            action = available_actions[np.argmax(q_values)]
            logger.debug(
                f"Chose greedy action {action} for state {state} (exploitation)"
            )
        return action

    def update_q(self, state: int, action: int, reward: float, next_state: int) -> None:
        available_actions = self.env.available_actions()
        best_next_action = available_actions[
            np.argmax(self.Q[next_state, available_actions])
        ]
        old_q = self.Q[state, action]
        self.Q[state, action] += self.alpha * (
            reward
            + self.gamma * self.Q[next_state, best_next_action]
            - self.Q[state, action]
        )
        logger.debug(
            f"Updated Q[{state}, {action}] from {old_q} to {self.Q[state, action]}"
        )

    def train(self, num_episodes: int) -> dict:
        logger.info(f"Starting training for {num_episodes} episodes")
        for episode in range(num_episodes):
            self.env.reset()
            state = self.env.state_id()
            total_reward = 0
            steps = 0

            while not self.env.is_game_over():
                action = self.choose_action(state)
                self.env.step(action)
                next_state = self.env.state_id()
                reward = self.env.score()
                total_reward += reward
                self.update_q(state, action, reward, next_state)
                state = next_state
                steps += 1

            logger.info(
                f"Episode {episode + 1} completed. Steps: {steps}, Total reward: {total_reward}"
            )

        logger.info("Training completed")
        return self.get_best_policy()

    def get_q_table(self) -> np.ndarray:
        logger.info("Returning Q-table")
        return self.Q

    def get_best_policy(self) -> dict:
        best_policy = {}
        for state in range(self.n_states):
            available_actions = self.env.available_actions()
            best_action = available_actions[np.argmax(self.Q[state, available_actions])]
            best_policy[state] = best_action
        logger.info("Generated best policy from Q-table")
        return best_policy


class DynaQ:
    def __init__(
        self,
        env: SecretEnv0 | SecretEnv1,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        dyna_planning_steps: int = 10,
    ):
        self.env = env
        self.num_states = env.num_states()
        self.num_actions = env.num_actions()
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Taux d'exploration
        self.dyna_planning_steps = dyna_planning_steps  # Nombre de simulations Dyna-Q

        self.QTable = np.zeros(
            (self.num_states, self.num_actions)
        )  # Table Q initialisée à zéro
        self.experiences: List[
            Tuple[int, int, int, float]
        ] = []  # Mémoire des expériences

        self.s: Optional[int] = None  # État courant
        self.action: Optional[int] = None  # Action courante
        logger.info(
            f"Initialized DynaQ with alpha={alpha}, gamma={gamma}, epsilon={epsilon}, dyna_planning_steps={dyna_planning_steps}"
        )

    def choose_action(self, state: int) -> int:
        available_actions = self.env.available_actions()
        if random.random() < self.epsilon:
            action = random.choice(available_actions)
            logger.debug(
                f"Chose random action {action} for state {state} (exploration)"
            )
        else:
            q_values = self.QTable[state, available_actions]
            action = available_actions[np.argmax(q_values)]
            logger.debug(
                f"Chose greedy action {action} for state {state} (exploitation)"
            )
        return action

    def update(self, s: int, a: int, s_prime: int, r: float) -> None:
        available_actions = self.env.available_actions()
        # Mise à jour de la table Q avec l'expérience réelle
        old_q = self.QTable[s, a]
        self.QTable[s, a] = (1 - self.alpha) * self.QTable[s, a] + self.alpha * (
            r + self.gamma * np.max(self.QTable[s_prime, available_actions])
        )
        logger.debug(f"Updated Q[{s}, {a}] from {old_q} to {self.QTable[s, a]}")

        # Stocker l'expérience
        self.experiences.append((s, a, s_prime, r))
        logger.debug(f"Stored experience: (s={s}, a={a}, s'={s_prime}, r={r})")

        # Planification Dyna-Q
        for i in range(self.dyna_planning_steps):
            if len(self.experiences) > 0:
                exp = random.choice(self.experiences)
                s_exp, a_exp, s_prime_exp, r_exp = exp
                available_actions_exp = self.env.available_actions()
                old_q = self.QTable[s_exp, a_exp]
                self.QTable[s_exp, a_exp] = (1 - self.alpha) * self.QTable[
                    s_exp, a_exp
                ] + self.alpha * (
                    r_exp
                    + self.gamma
                    * np.max(self.QTable[s_prime_exp, available_actions_exp])
                )
                logger.debug(
                    f"Dyna-Q step {i+1}: Updated Q[{s_exp}, {a_exp}] from {old_q} to {self.QTable[s_exp, a_exp]}"
                )

    def train(self, num_episodes: int) -> dict:
        logger.info(f"Starting training for {num_episodes} episodes")
        for episode in range(num_episodes):
            self.env.reset()
            self.s = self.env.state_id()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                self.action = self.choose_action(self.s)
                s_prime, r, done = self.step(self.s, self.action)
                self.update(self.s, self.action, s_prime, r)
                self.s = s_prime
                total_reward += r
                steps += 1

            logger.info(
                f"Episode {episode + 1} completed. Steps: {steps}, Total reward: {total_reward}"
            )

        logger.info("Training completed")
        return self.get_best_policy()

    def get_best_policy(self) -> dict:
        best_policy = {}
        for state in range(self.num_states):
            available_actions = self.env.available_actions()
            best_action = available_actions[
                np.argmax(self.QTable[state, available_actions])
            ]
            best_policy[state] = best_action
        logger.info("Generated best policy from Q-table")
        return best_policy

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        self.env.step(action)
        new_state = self.env.state_id()
        reward = self.env.score()
        done = self.env.is_game_over()
        logger.debug(
            f"Step: s={state}, a={action}, s'={new_state}, r={reward}, done={done}"
        )
        return new_state, reward, done
