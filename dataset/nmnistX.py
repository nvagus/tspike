import os
import pathlib

import click
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from joblib import Parallel, delayed

from nmnist import load_as_indices, indices_to_matrix, label_to_idx


def save_sampled_tmax(input_dir, label, output_dir, sample_rate, x_max, y_max, t_max, n_sample):
    for filename in tqdm(os.listdir(input_dir / label)):
        indices, _x_m, _y_m, _t_m = load_as_indices(
            input_dir / label / filename, sample_rate
        )
        matrix = indices_to_matrix(indices, x_max, y_max, t_max, "cpu")  # [on/off, x_max, y_max, t_max]
        result = np.zeros((2, x_max, y_max, n_sample)) # torch.zeros(2, x_max, y_max, t_max)
        sample_length = int(t_max / n_sample)
        for i in range(n_sample):
            result[:, :, :, i] = np.average(matrix[:, :, :, i*sample_length : (i+1) * sample_length], axis=3)

        file_dir = output_dir / label
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)

        file_name = file_dir / filename
        np.save(file_name, result, allow_pickle=True, fix_imports=True)


@click.command()
@click.option("-x", "--x-max", default=34)
@click.option("-y", "--y-max", default=34)
@click.option("-t", "--t-max", default=256)
@click.option("-t", "--n-sample", default=64)
@click.option("-s", "--sample-rate", default=1.0)
@click.option("-n", "--num-parallel", default=10)
@click.option("-i", "--input-path", default="../data/n-mnist/Train")
@click.option("-o", "--output-path", default="../data/n-mnist/TrainSPX")
def main(x_max, y_max, t_max, n_sample, sample_rate, num_parallel, input_path, output_path):
    input_dir = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_path, str(n_sample))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    

    Parallel(n_jobs=num_parallel)(
        delayed(save_sampled_tmax)(
            input_dir, label, output_dir, sample_rate, x_max, y_max, t_max, n_sample
        )
        for label in os.listdir(input_dir)
    )

    return 0


if __name__ == "__main__":
    exit(main())
