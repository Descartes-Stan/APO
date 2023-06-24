import os
from datasets import load_dataset

if __name__ == '__main__':
    dataset = "Liar"
    prompt = "Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other information."
    num_feedbacks = 4
    steps_per_gradient = 6
    beam_width = 4
    search_depth = 4
    time_steps = 10
    c = 1.0
    if dataset == "Liar":
        dataset = load_dataset("UKPLab/liar")
    command = f"python3 -u run_apo.py " \
              f"--dataset={dataset} " \
              f"--prompt={prompt} " \
              f"--num_feedbacks={num_feedbacks} " \
              f"--steps_per_gradient={steps_per_gradient} " \
              f"--beam_width={beam_width} " \
              f"--search_depth={search_depth} " \
              f"--time_steps={time_steps} " \
              f"--c={c} " \

    print(command)
    os.system(command)