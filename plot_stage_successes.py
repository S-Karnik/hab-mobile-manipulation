import itertools
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

data_baseline_1 = """2023-05-06 23:17:48,166 Average episode stage_success.open_0: 1.0000
2023-05-06 23:17:48,166 Average episode stage_success.pick_0: 0.9800
2023-05-06 23:17:48,166 Average episode stage_success.place_0: 0.9400
2023-05-06 23:17:48,166 Average episode stage_success.close_0: 0.9100
2023-05-06 23:17:48,166 Average episode stage_success.open_1: 0.9100
2023-05-06 23:17:48,166 Average episode stage_success.pick_1: 0.8900
2023-05-06 23:17:48,166 Average episode stage_success.place_1: 0.8100
2023-05-06 23:17:48,167 Average episode stage_success.close_1: 0.3100"""

data_baseline_2 = """2023-05-17 16:47:40,387 Average episode stage_success.open_0: 0.9800
2023-05-17 16:47:40,387 Average episode stage_success.pick_0: 0.9600
2023-05-17 16:47:40,387 Average episode stage_success.place_0: 0.9200
2023-05-17 16:47:40,387 Average episode stage_success.close_0: 0.7500
2023-05-17 16:47:40,387 Average episode stage_success.open_1: 0.7500
2023-05-17 16:47:40,387 Average episode stage_success.pick_1: 0.7100
2023-05-17 16:47:40,387 Average episode stage_success.place_1: 0.7000
2023-05-17 16:47:40,387 Average episode stage_success.close_1: 0.4700"""

data_baseline_3 = """2023-06-01 04:23:55,089 Average episode stage_success.open_0: 1.0000
2023-06-01 04:23:55,089 Average episode stage_success.pick_0: 0.9800
2023-06-01 04:23:55,089 Average episode stage_success.place_0: 0.9400
2023-06-01 04:23:55,089 Average episode stage_success.close_0: 0.8400
2023-06-01 04:23:55,089 Average episode stage_success.open_1: 0.8400
2023-06-01 04:23:55,089 Average episode stage_success.pick_1: 0.8300
2023-06-01 04:23:55,089 Average episode stage_success.place_1: 0.8000
2023-06-01 04:23:55,089 Average episode stage_success.close_1: 0.6900"""

pattern = r"stage_success\.(\w+_\d+):\s(\d+\.\d+)"

def plot_success(env_name, all_data, all_data_names):
    colors = iter(mcolors.BASE_COLORS)
    for data, data_name in zip(all_data, all_data_names):
        matches = re.findall(pattern, data)

        actions = []
        success_rates = []

        for match in matches:
            actions.append(match[0])
            success_rates.append(float(match[1]))

        plt.plot(actions, success_rates, f'{next(colors)}-', label=data_name)
    plt.xlabel('Actions')
    plt.ylabel('Success Rate')
    plt.title('Success Rate of Actions')
    plt.xticks(rotation=45)
    plt.legend()
    plt.ylim(0, 1)  # Set the y-axis limits between 0 and 1
    plt.grid(True)
    plt.savefig(f"plots/{env_name}.png")

import json
close_0_fails = {(False, False): 0, (True, False): 0, (False, True): 0}
true_false_combos = list(itertools.product([True, False], repeat=4))
close_1_fails = {tuple(true_false_combos[i]): 0 for i in range(len(true_false_combos))}
def compute_num_future_place_fails(stage_successes_filepath, result_filepath):
    stage_successes = {}
    with open(stage_successes_filepath, 'rb') as f:
        stage_successes = json.load(f)
    with open(result_filepath, 'rb') as f:
        results = json.load(f)
        num_successes = []
        for result in results:
            num_successes.append(np.sum([result[ss] for ss in result if 'stage_success' in ss]))
    for i, episode in enumerate(stage_successes):
        if not all(episode['close_0']) and num_successes[i] == 3:
            close_0_fails[tuple(episode['close_0'])] += 1
        if not all(episode['close_1']) and num_successes[i] == 7:
            close_1_fails[tuple(episode['close_1'])] += 1
    import pdb; pdb.set_trace()

# stage_successes_filepath = "data/results/rearrange/composite/set_table/mr/stage_success.json"
# result_filepath = "data/results/rearrange/composite/set_table/mr/result.json"
# compute_num_future_place_fails(stage_successes_filepath, result_filepath)

plot_success("SetTable", [data_baseline_1, data_baseline_2, data_baseline_3], ["Baseline", "Bilinear Actor-Critic Skills", "Baseline + Stable Place Skill"])


