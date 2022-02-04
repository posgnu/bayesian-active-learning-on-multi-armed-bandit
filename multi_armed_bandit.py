import numpy as np

class MultiArmedBandit:
    def __init__(self, K: int, theta_list: list) -> None:
        assert len(theta_list) == K
        for theta in theta_list:
            assert 0 <= theta <= 1

        self.K = K
        self.theta_list = theta_list

    def pull_arm(self, k: int) -> int:
        return np.random.binomial(1, self.theta_list[k])
