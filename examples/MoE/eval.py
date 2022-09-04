import os

import subprocess


batch_sizes = [1]

mp_levels = [1, 2]

model_sizes = ["125M", "350M", "760M", "1.3B", "2.7B", "6.7B", "13B", "175B"]

for model_size in model_sizes:
    dirname = f"result{model_size}"
    subprocess.run(f"mkdir -p {dirname}", shell=True, check=True)

    for mp_level in mp_levels:

        visible_devices = ""

        for i in range(mp_level):

            visible_devices += f"{i},"

        visible_devices = visible_devices[:-1]

        for batch_size in batch_sizes:

            filename = f"{dirname}/mp{mp_level}-g{mp_level}-b{batch_size}.txt"

            f = open(filename, "w")

            cmd = f"CUDA_VISIBLE_DEVICES={visible_devices} bash ds_inf_gpt_125M_MoE64.sh {batch_size} {mp_level} {model_size}"

            # print(cmd)

            subprocess.run(cmd, shell=True, check=True, stdout=f)
