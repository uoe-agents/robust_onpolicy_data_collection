import torch
import numpy as np
from torch.nn.functional import one_hot, softmax

from experiments.multibandit import MultiBandit


def create_GW_policies():
    basic_policy = torch.load(f'policy_models/model_GridWorld_5000.pt')

    # copy for different env.
    for mean_factor in range(1, 11):
        torch.save(basic_policy, f'policy_models/model_GridWorldM{mean_factor}S1_5000.pt')
    for scale_factor in range(1, 11):
        torch.save(basic_policy, f'policy_models/model_GridWorldM1S{scale_factor}_5000.pt')

    weight = basic_policy.model[1][0].weight.data
    one_hot_weight = one_hot(weight.argmax(1), num_classes=weight.shape[1])
    for epsilon in range(11):
        epsilon /= 10
        if epsilon == 0:
            w = 1000.
        else:
            w = np.log((4 - 3 * epsilon) / epsilon)

        basic_policy.model[1][0].weight.data = one_hot_weight * w
        print(softmax(one_hot_weight * w, 1)[0])
        torch.save(basic_policy, f'policy_models/model_GridWorldM1S0_{int(epsilon * 10)}.pt')


def create_MB_policies():
    basic_policy = torch.load(f'policy_models/model_MultiBandit_5000.pt')

    # copy for different env.
    for mean_factor in range(1, 11):
        torch.save(basic_policy, f'policy_models/model_MultiBanditM{mean_factor}S1_5000.pt')
    for scale_factor in range(1, 11):
        torch.save(basic_policy, f'policy_models/model_MultiBanditM1S{scale_factor}_5000.pt')

    # eps_policy for original env.
    env = MultiBandit()
    n_action = env.means.shape[0]
    optimal_action = env.means.argmax(0).item()
    one_hot_weight = one_hot(torch.Tensor([optimal_action]).long(), num_classes=n_action)
    for epsilon in range(11):
        epsilon /= 10
        if epsilon == 0:
            w = 1000.
        else:
            w = np.log(n_action / epsilon - n_action + 1)

        basic_policy.model[1][0].weight.data = one_hot_weight * w
        print(softmax(one_hot_weight * w, 1)[0].max())
        torch.save(basic_policy, f'policy_models/model_MultiBanditM1S1_{int(epsilon * 10)}.pt')


if __name__ == '__main__':
    create_GW_policies()
    create_MB_policies()
