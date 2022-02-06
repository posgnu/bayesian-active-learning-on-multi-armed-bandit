from multi_armed_bandit import MultiArmedBandit
from agents import RandomAgent, ThompsonAgent, GreedyAgent
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")


def learn(M, K, theta_list, bandit_machine, N):
    num_success_R = np.zeros(N)
    num_success_TS = np.zeros(N)
    num_success_G = np.zeros(N)

    for num_run in range(M):
        R = RandomAgent(K)
        TS = ThompsonAgent(K)
        G = GreedyAgent(K)
        for num_trial in range(N):
            # Since the choice is not deterministic when there is a tie we need a temporary variable
            R_choice = R.trial()
            R.update(R_choice, bandit_machine.pull_arm(R_choice))
            num_success_R[num_trial] += (
                1 if np.argmax(theta_list) == R.best_arm() else 0
            )

            TS_choice = TS.trial()
            TS.update(TS_choice, bandit_machine.pull_arm(TS_choice))
            num_success_TS[num_trial] += (
                1 if np.argmax(theta_list) == TS.best_arm() else 0
            )

            G_choice = G.trial()
            G.update(G_choice, bandit_machine.pull_arm(G_choice))
            num_success_G[num_trial] += (
                1 if np.argmax(theta_list) == G.best_arm() else 0
            )

            # print(f"run {num_run} - trial {num_trial} - R {R_choice} - TS {TS_choice} - G {G_choice}")

    return num_success_R, num_success_TS, num_success_G


def first_experiment(axs, M, K):
    theta_list = [0.9, 0.8] + [0.5 for _ in range(8)]
    bandit_machine = MultiArmedBandit(K, theta_list)
    N = 1000

    num_success_R, num_success_TS, num_success_G = learn(
        M,
        K,
        theta_list,
        bandit_machine,
        N,
    )

    x = list(range(1, N + 1))
    axs[0].plot(x, num_success_R / M, lw=2, ls="-", alpha=0.5, label="Random")
    axs[0].plot(
        x, num_success_TS / M, lw=2, ls="-", alpha=0.5, label="Thompson Sampling"
    )
    axs[0].plot(x, num_success_G / M, lw=2, ls="-", alpha=0.5, label="Greedy")
    axs[0].legend()
    axs[0].set_title("Experiment 1")
    axs[0].set_xlabel("trials")
    axs[0].set_ylabel("success rate")


def second_experiment(axs, M, K):
    theta_list = [0.9, 0.88] + [0.5 for _ in range(8)]
    bandit_machine = MultiArmedBandit(K, theta_list)
    N = 3000

    num_success_R, num_success_TS, num_success_G = learn(
        M, K, theta_list, bandit_machine, N
    )

    x = list(range(1, N + 1))
    axs[1].plot(x, num_success_R / M, lw=2, ls="-", alpha=0.5, label="Random")
    axs[1].plot(
        x, num_success_TS / M, lw=2, ls="-", alpha=0.5, label="Thompson Sampling"
    )
    axs[1].plot(x, num_success_G / M, lw=2, ls="-", alpha=0.5, label="Greedy")
    axs[1].legend()
    axs[1].set_title("Experiment 2")
    axs[1].set_xlabel("trials")
    axs[1].set_ylabel("success rate")


def third_experiment(axs, M, K):
    theta_list = [0.9] + [0.85 for _ in range(9)]
    bandit_machine = MultiArmedBandit(K, theta_list)
    N = 3000

    num_success_R, num_success_TS, num_success_G = learn(
        M, K, theta_list, bandit_machine, N
    )

    x = list(range(1, N + 1))
    axs[2].plot(x, num_success_R / M, lw=2, ls="-", alpha=0.5, label="Random")
    axs[2].plot(
        x, num_success_TS / M, lw=2, ls="-", alpha=0.5, label="Thompson Sampling"
    )
    axs[2].plot(x, num_success_G / M, lw=2, ls="-", alpha=0.5, label="Greedy")
    axs[2].legend()
    axs[2].set_title("Experiment 3")
    axs[2].set_xlabel("trials")
    axs[2].set_ylabel("success rate")


if __name__ == "__main__":
    fig, axs = plt.subplots(3)
    M = 500
    K = 10

    first_experiment(axs, M, K)

    second_experiment(axs, M, K)

    third_experiment(axs, M, K)

    plt.tight_layout()
    plt.show()
