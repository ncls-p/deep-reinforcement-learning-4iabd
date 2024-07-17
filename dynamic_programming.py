import logging
import pickle  # Add import for pickle
from typing import Dict

import numpy as np

from secret_envs_wrapper import SecretEnv0, SecretEnv1

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DynamicProgramming:
    def __init__(
        self,
        env: SecretEnv0 | SecretEnv1,
        gamma: float,
        theta: float,
        max_iterations: int,
    ):
        self.env: SecretEnv0 | SecretEnv1 = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.state_ids = range(self.env.num_states())
        self.actions = range(self.env.num_actions())
        self.V = np.zeros(self.env.num_states())
        logger.info(
            f"Initialized DynamicProgramming with gamma={gamma}, theta={theta}, max_iterations={max_iterations}"
        )

    def one_step_lookahead(self, state: int) -> np.ndarray:
        action_values = np.zeros(self.env.num_actions())
        for action in self.actions:
            if self.env.is_forbidden(action):
                continue
            value = 0
            for s_next in self.state_ids:
                for r in range(self.env.num_rewards()):
                    p = self.env.p(state, action, s_next, r)
                    value += p * (self.env.reward(r) + self.gamma * self.V[s_next])
            action_values[action] = value
            logger.debug(f"State {state}, Action {action}: Value {value}")
        return action_values

    def print_results(self, policy: Dict[int, int]):
        logger.info("Value Function:")
        for state in self.state_ids:
            logger.info(f"State {state}: {self.V[state]}")
        logger.info("Policy:")
        for state in self.state_ids:
            logger.info(f"State {state}: {policy[state]}")

    def get_best_policy(self) -> Dict[int, int]:
        policy = {}
        for state in self.state_ids:
            action_values = self.one_step_lookahead(state)
            best_action = int(np.argmax(action_values))
            if not self.env.is_forbidden(best_action):
                policy[state] = best_action
            else:
                # Choose the best non-forbidden action
                for action in np.argsort(action_values)[::-1]:
                    if not self.env.is_forbidden(action):
                        policy[state] = int(action)
                        break
            logger.debug(f"State {state}: Best Action {policy[state]}")
        return policy


class PolicyIteration(DynamicProgramming):
    def __init__(self, env_type, gamma: float, theta: float, max_iterations: int):
        super().__init__(env_type, gamma, theta, max_iterations)
        self.policy = np.random.choice(self.actions, size=self.env.num_states())
        logger.info("Initialized PolicyIteration")

    def policy_evaluation(self):
        logger.info("Starting policy evaluation")
        iteration = 0
        while True:
            delta = 0
            for state in self.state_ids:
                v = self.V[state]
                action = self.policy[state]
                if self.env.is_forbidden(action):
                    continue
                value = 0
                for s_next in self.state_ids:
                    for r in range(self.env.num_rewards()):
                        p = self.env.p(state, action, s_next, r)
                        value += p * (self.env.reward(r) + self.gamma * self.V[s_next])
                self.V[state] = value
                delta = max(delta, abs(v - self.V[state]))
            iteration += 1
            if delta < self.theta:
                break
        logger.info("Policy evaluation completed")

    def policy_improvement(self) -> bool:
        logger.info("Starting policy improvement")
        policy_stable = True
        for state in self.state_ids:
            old_action = self.policy[state]
            action_values = self.one_step_lookahead(state)
            new_action = int(np.argmax(action_values))
            if self.env.is_forbidden(new_action):
                continue
            self.policy[state] = new_action
            if old_action != new_action:
                policy_stable = False
        logger.info(f"Policy improvement completed. Policy stable: {policy_stable}")
        return policy_stable

    def run(self) -> Dict[int, int]:
        logger.info("Starting PolicyIteration run")
        for i in range(self.max_iterations):
            logger.info(f"Iteration {i+1}")
            self.policy_evaluation()
            if self.policy_improvement():
                logger.info(f"PolicyIteration converged after {i+1} iterations")
                break
        policy = self.get_best_policy()
        self.print_results(policy)
        return policy


class ValueIteration(DynamicProgramming):
    def run(self) -> Dict[int, int]:
        logger.info("Starting ValueIteration run")
        for i in range(self.max_iterations):
            delta = 0
            for state in self.state_ids:
                v = self.V[state]
                action_values = self.one_step_lookahead(state)
                self.V[state] = max(action_values)
                delta = max(delta, abs(v - self.V[state]))
                logger.debug(
                    f"Iteration {i+1}, State {state}: Value {self.V[state]}, Delta: {delta}"
                )
            logger.info(f"Iteration {i+1}: Delta {delta}")
            if delta < self.theta:
                logger.info(f"ValueIteration converged after {i+1} iterations")
                break
        policy = self.get_best_policy()
        logger.info("ValueIteration completed")
        self.print_results(policy)
        return policy
