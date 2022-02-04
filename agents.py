import random
import numpy as np


class Agent:
    def __init__(self, K: int, prior="flat") -> None:
        self.K = K
        if prior == "flat":
            self.alpha_list = np.ones(K)
            self.beta_list = np.ones(K)
        else:
            raise NotImplemented
        self.mpe_list = self.alpha_list / (self.alpha_list + self.beta_list)

    def trial(self) -> int:
        raise NotImplemented

    def update(self, k: int, reward: int) -> None:
        assert reward == 0 or reward == 1
        assert k < self.K

        self.alpha_list[k] += reward
        self.beta_list[k] += 1 - reward
        self.mpe_list[k] = self.alpha_list[k] / (self.alpha_list[k] + self.beta_list[k])

        return

    def best_arm(self) -> int:
        return random.choice(
            [i for i, x in enumerate(self.mpe_list) if x == max(self.mpe_list)]
        )


class RandomAgent(Agent):
    def __init__(self, K: int, prior="flat") -> None:
        super().__init__(K, prior)

    def trial(self) -> int:
        return random.choice(range(self.K))


class ThompsonAgent(Agent):
    def __init__(self, K: int, prior="flat") -> None:
        super().__init__(K, prior)

    def trial(self) -> int:
        samples = [
            np.random.beta(alpha, beta)
            for alpha, beta in zip(self.alpha_list, self.beta_list)
        ]
        return random.choice([i for i, x in enumerate(samples) if x == max(samples)])


class GreedyAgent(Agent):
    def __init__(self, K: int, prior="flat") -> None:
        super().__init__(K, prior)

    def trial(self) -> int:
        return random.choice(
            [i for i, x in enumerate(self.mpe_list) if x == max(self.mpe_list)]
        )
