import numpy as np
from typing import List, Tuple


class LineWorld:
    def __init__(self):
        self.agent_pos = 2

    # Uniquement pour le MonteCarloES
    def from_random_state() -> "LineWorld":
        env = LineWorld()
        env.agent_pos = np.random.randint(1, 4)
        return env

    def available_actions(self) -> List[int]:
        if self.agent_pos in [1, 2, 3]:
            return [0, 1]  # 0: left, 1: right
        return []

    def is_game_over(self) -> bool:
        return True if self.agent_pos in [0, 4] else False

    def state_id(self) -> int:
        return self.agent_pos

    def num_states(self):
        self.state_id()

    def num_actions(self):
        self.available_actions()

    def step(self, action: int):
        assert not self.is_game_over()
        assert action in self.available_actions()

        if action == 0:
            self.agent_pos -= 1
        else:
            self.agent_pos += 1

    def score(self) -> float:
        if self.agent_pos == 0:
            return -1.0
        if self.agent_pos == 4:
            return 1.0
        return 0.0

    def display(self):
        for i in range(5):
            print("X" if self.agent_pos == i else "_", end="")
        print()

    def reset(self):
        self.agent_pos = 2


class GridWorld:
    # 20;21;22;23;24
    # 15;16;17;18;19
    # 10;11;12;13;14
    # 5 ;6 ;7 ;8 ;9
    # 0 ;1 ;2 ;3 ;4
    def __init__(self):
        self.agent_pos = 12  # Starting in position 12

    # Uniquement pour le MonteCarloES
    @staticmethod
    def from_random_state() -> "GridWorld":
        env = GridWorld()
        env.agent_pos = np.random.choice([6, 16, 17, 11, 12, 13, 7, 8])
        return env

    def available_actions(self) -> List[int]:
        actions = []
        row, col = self._get_coordinates(self.agent_pos)

        if col > 0:  # Can move left
            actions.append(0)
        if col < 4:  # Can move right
            actions.append(1)
        if row > 0:  # Can move up
            actions.append(2)
        if row < 4:  # Can move down
            actions.append(3)

        return actions

    def is_game_over(self) -> bool:
        return self.agent_pos not in [6, 16, 17, 11, 12, 13, 7, 8, 18]

    def state_id(self) -> int:
        return self.agent_pos

    def num_states(self):
        self.state_id()

    def num_actions(self):
        self.available_actions()

    def step(self, action: int):
        assert not self.is_game_over()
        assert action in self.available_actions()

        row, col = self._get_coordinates(self.agent_pos)

        if action == 0:  # Move left
            col -= 1
        elif action == 1:  # Move right
            col += 1
        elif action == 2:  # Move up
            row -= 1
        elif action == 3:  # Move down
            row += 1

        self.agent_pos = self._get_position(row, col)

    def score(self) -> float:
        if self.agent_pos == 18:
            return 1.0
        elif self.agent_pos in [6]:
            return -1.0
        else:
            return 0.0

    def display(self):
        for i in range(5):
            for j in range(5):
                pos = self._get_position(i, j)
                if self.agent_pos == pos:
                    print("X", end=" ")
                elif pos == 18:
                    print("R", end=" ")  # Reward
                elif pos in [6, 16, 17, 11, 12, 13, 7, 8]:
                    print("N", end=" ")  # Neutral
                else:
                    print("_", end=" ")  # Game over
            print()

    def reset(self):
        self.agent_pos = 6

    def _get_coordinates(self, pos: int) -> Tuple[int, int]:
        row = pos // 5
        col = pos % 5
        return row, col

    def _get_position(self, row: int, col: int) -> int:
        return row * 5 + col


class TwoRoundRockPaperScissors:
    def __init__(self):
        self.reset()

    # Static method to generate a random state
    @staticmethod
    def from_random_state() -> "TwoRoundRockPaperScissors":
        env = TwoRoundRockPaperScissors()
        env.round = 1
        env.agent_choices[0] = np.random.choice([0, 1, 2])  # Agent's first-round choice
        env.opponent_choices[0] = np.random.choice(
            [0, 1, 2]
        )  # Opponent's first-round choice
        return env

    # Available actions for Rock, Paper, Scissors
    def available_actions(self) -> List[int]:
        return [0, 1, 2]  # 0: Rock, 1: Paper, 2: Scissors

    # Check if the game is over (after 2 rounds)
    def is_game_over(self) -> bool:
        return self.round > 2

    # Get the current state ID (round and agent's first choice)
    def state_id(self) -> int:
        return self.round * 10 + (self.agent_choices[0] if self.round > 1 else 0)

    def num_states(self):
        self.state_id()

    def num_actions(self):
        self.available_actions()

    # Simulate a step in the game
    def step(self, action: int):
        assert not self.is_game_over()
        assert action in self.available_actions()

        self.agent_choices[self.round - 1] = action

        if self.round == 1:
            self.opponent_choices[self.round - 1] = np.random.choice([0, 1, 2])
        else:
            self.opponent_choices[self.round - 1] = self.agent_choices[0]

        self.round += 1

    # Calculate the score after each round
    def score(self) -> float:
        if self.round == 1:  # First round
            return self._round_score(self.agent_choices[0], self.opponent_choices[0])
        elif self.round == 2:  # Second round
            first_round_score = self._round_score(
                self.agent_choices[0], self.opponent_choices[0]
            )
            second_round_score = self._round_score(
                self.agent_choices[1], self.opponent_choices[1]
            )
            return first_round_score + second_round_score
        return 0.0

    # Reset the game to the initial state
    def reset(self):
        self.round = 1
        self.agent_choices = [-1, -1]
        self.opponent_choices = [-1, -1]

    # Helper method to calculate the score of a single round
    def _round_score(self, agent_choice: int, opponent_choice: int) -> float:
        if agent_choice == opponent_choice:
            return 0.0
        if (
            (agent_choice == 0 and opponent_choice == 2)
            or (agent_choice == 1 and opponent_choice == 0)
            or (agent_choice == 2 and opponent_choice == 1)
        ):
            return 1.0
        return -1.0

    # Display the game status
    def display(self):
        choices = ["Rock", "Paper", "Scissors"]
        if self.round > 1:
            print(
                f"Round 1: Agent chose {choices[self.agent_choices[0]]}, Opponent chose {choices[self.opponent_choices[0]]}"
            )
        if self.round > 2:
            print(
                f"Round 2: Agent chose {choices[self.agent_choices[1]]}, Opponent chose {choices[self.opponent_choices[1]]}"
            )
        print(f"Score: {self.score()}")


class MontyHallLvl1:
    def __init__(self):
        self.winning_door = np.random.choice(["A", "B", "C"])
        self.first_choice = None
        self.revealed_door = None
        self.second_choice = None
        self.state = (
            "CHOOSE_FIRST"  # Possible states: 'CHOOSE_FIRST', 'CHOOSE_SECOND', 'END'
        )

    @staticmethod
    def from_random_state() -> "MontyHallLvl1":
        env = MontyHallLvl1()
        env.winning_door = np.random.choice(["A", "B", "C"])
        return env

    def available_actions(self) -> List[str]:
        if self.state == "CHOOSE_FIRST":
            return ["A", "B", "C"]
        elif self.state == "CHOOSE_SECOND":
            return ["STAY", "SWITCH"]
        return []

    def is_game_over(self) -> bool:
        return self.state == "END"

    def state_id(self) -> str:
        return f"{self.state}:{self.first_choice}:{self.revealed_door}"

    def num_states(self):
        self.state_id()

    def num_actions(self):
        self.available_actions()

    def step(self, action: str):
        assert not self.is_game_over()
        assert action in self.available_actions()

        if self.state == "CHOOSE_FIRST":
            self.first_choice = action
            self.revealed_door = self._reveal_door()
            self.state = "CHOOSE_SECOND"
        elif self.state == "CHOOSE_SECOND":
            if action == "STAY":
                self.second_choice = self.first_choice
            else:
                self.second_choice = self._switch_choice()
            self.state = "END"

    def score(self) -> float:
        if self.state != "END":
            return 0.0
        return 1.0 if self.second_choice == self.winning_door else 0.0

    def display(self):
        doors = ["A", "B", "C"]
        display_str = ""
        for door in doors:
            if door == self.second_choice:
                display_str += "X" if door == self.winning_door else "O"
            else:
                display_str += "_"
        print(display_str)

    def reset(self):
        self.__init__()

    def _reveal_door(self) -> str:
        doors = ["A", "B", "C"]
        doors.remove(self.first_choice)
        if self.winning_door in doors:
            doors.remove(self.winning_door)
        return doors[0]

    def _switch_choice(self) -> str:
        doors = ["A", "B", "C"]
        doors.remove(self.first_choice)
        doors.remove(self.revealed_door)
        return doors[0]


class MontyHallLvl2:
    def __init__(self):
        self.doors = ["A", "B", "C", "D", "E"]
        self.winning_door = np.random.choice(self.doors)
        self.choices = []
        self.revealed_doors = []
        self.state = "CHOOSE_FIRST"  # Possible states: 'CHOOSE_FIRST', 'CHOOSE_SECOND', 'CHOOSE_THIRD', 'CHOOSE_FOURTH', 'END'

    @staticmethod
    def from_random_state() -> "MontyHallLvl2":
        env = MontyHallLvl2()
        env.winning_door = np.random.choice(env.doors)
        return env

    def available_actions(self) -> List[str]:
        if self.state.startswith("CHOOSE"):
            return [
                door
                for door in self.doors
                if door not in self.choices and door not in self.revealed_doors
            ]
        return []

    def is_game_over(self) -> bool:
        return self.state == "END"

    def state_id(self) -> str:
        return f"{self.state}:{self.choices}:{self.revealed_doors}"

    def num_states(self):
        self.state_id()

    def num_actions(self):
        self.available_actions()

    def step(self, action: str):
        assert not self.is_game_over()
        assert action in self.available_actions()

        self.choices.append(action)

        if self.state == "CHOOSE_FIRST":
            self.state = "CHOOSE_SECOND"
        elif self.state == "CHOOSE_SECOND":
            self.state = "CHOOSE_THIRD"
        elif self.state == "CHOOSE_THIRD":
            self.state = "CHOOSE_FOURTH"
        elif self.state == "CHOOSE_FOURTH":
            self.revealed_doors = self._reveal_doors()
            self.state = "CHOOSE_FINAL"
        elif self.state == "CHOOSE_FINAL":
            self.state = "END"

    def score(self) -> float:
        if self.state != "END":
            return 0.0
        return 1.0 if self.choices[-1] == self.winning_door else 0.0

    def display(self):
        for door in self.doors:
            if door in self.choices:
                if door == self.winning_door:
                    print("X", end="")
                else:
                    print("O", end="")
            else:
                print("_", end="")
        print()

    def reset(self):
        self.__init__()

    def _reveal_doors(self) -> List[str]:
        remaining_doors = [door for door in self.doors if door not in self.choices]
        non_winning_doors = [
            door for door in remaining_doors if door != self.winning_door
        ]
        np.random.shuffle(non_winning_doors)
        return non_winning_doors[: len(non_winning_doors) - 1]
