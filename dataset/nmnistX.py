import os
import pathlib

import click
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from joblib import Parallel, delayed

from nmnist import load_as_indices, indices_to_matrix, label_to_idx


def save_sampled_tmax(input_dir, label, output_dir, sample_rate, x_max, y_max, t_max):
    for filename in tqdm(os.listdir(input_dir / label)):
        indices, _x_m, _y_m, _t_m = load_as_indices(
            input_dir / label / filename, sample_rate
        )
        matrix = indices_to_matrix(indices, x_max, y_max, t_max, "cpu")

        file_dir = output_dir / label
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)

        file_name = file_dir / filename
        np.save(file_name, matrix, allow_pickle=True, fix_imports=True)


@click.command()
@click.option("-x", "--x-max", default=34)
@click.option("-y", "--y-max", default=34)
@click.option("-t", "--t-max", default=30)
@click.option("-s", "--sample-rate", default=1.0)
@click.option("-n", "--num-parallel", default=10)
@click.option("-i", "--input-path", default="../data/n-mnist/Train")
@click.option("-o", "--output-path", default="../data/n-mnist/TrainSPX")
def main(x_max, y_max, t_max, sample_rate, num_parallel, input_path, output_path):
    input_dir = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_path, str(t_max))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    

    Parallel(n_jobs=num_parallel)(
        delayed(save_sampled_tmax)(
            input_dir, label, output_dir, sample_rate, x_max, y_max, t_max
        )
        for label in os.listdir(input_dir)
    )

    return 0


if __name__ == "__main__":
    exit(main())
