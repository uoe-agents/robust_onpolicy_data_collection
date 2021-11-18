import os
from commons import save, load


def merge(m_result: dict, s_result: dict):
    for k, v in s_result.items():
        if k not in m_result:
            m_result[k] = v
        else:
            if k == 'eval_steps':
                continue
            if type(v) is dict:
                merge(m_result[k], v)
            if type(v) is list:
                m_result[k].extend(v)


merge_needed = set()
result_root = 'results'
seed_root = 'results/seeds'

for file in os.listdir(seed_root):
    merge_needed.add('_'.join(file.split('_')[:-2]))

merge_needed = sorted(merge_needed)
print(merge_needed)

log = ''
for file in merge_needed:
    result_file = os.path.join(result_root, file)

    try:
        main_result = load(result_file)
    except:
        main_result = {'seeds': []}

    changed = False
    for seed in range(1000):
        if seed in main_result['seeds']:
            continue

        seed_file = os.path.join(seed_root, file) + f'_{seed}_{seed + 1}'
        if not os.path.exists(seed_file):
            continue
        try:
            seed_result = load(seed_file)
        except:
            import traceback

            traceback.print_exc()
            continue

        main_result['seeds'].append(seed)
        merge(main_result, seed_result)
        changed = True

    if changed:
        save(result_file, main_result)
        log += f"{file}: {len(main_result['seeds'])} seeds.\n"

print(log)
# python experiments/merge_results.py
