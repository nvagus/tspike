import os
import pathlib

import click
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_as_indices(filepath, sample_rate):
    raw = np.fromfile(filepath, dtype=np.uint8)
    x = raw[0::5]
    y = raw[1::5]
    p = raw[2::5] & 128 >> 7
    t = ((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | (raw[4::5])
    limit = y != 240
    x = x[limit]
    y = y[limit]
    p = p[limit]
    t = np.floor(t[limit] / sample_rate).astype(np.uint32)
    return np.vstack([t.astype(np.uint8), p, x, y]), x.max(), y.max(), t.max()


def indices_to_matrix(mat, x_max, y_max, t_max):
    result = torch.zeros(2, x_max, y_max, t_max, dtype=torch.int32)
    for t, p, x, y in mat.T:
        if t >= t_max:
            continue
        result[p, x, y, t] = 1
    return result


def label_to_idx(label):
    return int(label)


class NMnistSampled(Dataset):
    def __init__(self, root_dir, x_max, y_max, t_max):
        self.dir = pathlib.Path(root_dir)
        self.files = os.listdir(self.dir)
        self.x_max = x_max
        self.y_max = y_max
        self.t_max = t_max

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.dir / self.files[idx])
        return {
            'data': indices_to_matrix(data['indices'], self.x_max, self.y_max, self.t_max),
            'label': label_to_idx(data['label'])
        }


@click.command()
@click.option('-x', '--x-max', default=34)
@click.option('-y', '--y-max', default=34)
@click.option('-t', '--t-max', default=256)
@click.option('-s', '--sample-rate', default=1.)
@click.option('-i', '--input-path', default='../data/n-mnist/Train')
@click.option('-o', '--output-path', default='../data/n-mnist/TrainSP')
def main(
    x_max, y_max, t_max,
    sample_rate, input_path, output_path
):
    input_dir = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for label in os.listdir(input_dir):
        for filename in tqdm(os.listdir(input_dir / label)):
            indices, x_m, y_m, t_m = load_as_indices(input_dir / label / filename, sample_rate)
            x_max = max(x_max, x_m)
            y_max = max(y_max, y_m)
            t_max = max(t_max, t_m)
            np.savez(output_dir/f'{label}-{filename}.npz', indices=indices, label=label)

    print('x_max: ', x_max)
    print('y_max: ', y_max)
    print('t_max: ', t_max)
    return 0


if __name__ == '__main__':
    exit(main())
