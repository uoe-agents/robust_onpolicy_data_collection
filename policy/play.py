from policy.reinforce import Configs, REINFORCE
from commons import mc_estimation_true_value

policy_name = 'GridWorld_10'
# policy_name = 'GridWorld_6000'
# policy_name = 'CartPoleContinuous_4000'

exp_name, _ = policy_name.split('_')
env, max_time_step, gamma, _, save_episodes = Configs[exp_name]
actor = REINFORCE(env, gamma, file=f'policy_models/model_{policy_name}.pt', max_time_step=max_time_step, device='cpu')

for i in range(10):
    print(mc_estimation_true_value(env, max_time_step, actor, gamma, 1, show=False, render=True))
