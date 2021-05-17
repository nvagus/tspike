from typing import ItemsView
import click
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import NMnistSampled
from tnn import SpikesTracer, FullColumn


class Interrupter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            return True
        return exc_type is None


@click.command()
# context
@click.option('-g', '--gpu', default=0)
@click.option('-e', '--epochs', default=1)
@click.option('-b', '--batch', default=32)
@click.option('-x', '--x-max', default=34)
@click.option('-y', '--y-max', default=34)
@click.option('-t', '--t-max', default=256)
# spiking control
@click.option('--dense', default=0.04)
@click.option('--theta', default=0)
@click.option('--forced-dep', default=1024)
@click.option('--w-init', default=0.30)
@click.option('--bias', default=0.30)
@click.option('--step', default=8)
@click.option('--leak', default=64)
# model structure
@click.option('--neurons', default=1)
@click.option('--winners', default=1)
# learning parameters
@click.option('--decay', default=0.9999)
# paths
@click.option('--train-path', default='data/n-mnist/TrainSP')
@click.option('--test-path', default='data/n-mnist/TestSP')
@click.option('--model-path', default='model/n-mnist-fc')
def main(
    gpu, batch, epochs,
    x_max, y_max, t_max,
    dense, theta, forced_dep, w_init, bias,
    step, leak,
    neurons, winners,
    decay,
    train_path, test_path, model_path,
    **kwargs
):
    if torch.cuda.is_available():
        dev = f'cuda:{gpu}'
    else:
        dev = 'cpu'
    device = torch.device(dev)

    train_data_loader = DataLoader(NMnistSampled(
        train_path, x_max, y_max, t_max, device=device), shuffle=True, batch_size=batch)
    test_data_loader = DataLoader(NMnistSampled(
        test_path, x_max, y_max, t_max, device=device), batch_size=batch)

    model = FullColumn(
        x_max * y_max, neurons, input_channel=2, output_channel=10,
        step=step, leak=leak, bias=bias, winners=winners,
        fodep=forced_dep, w_init=w_init, theta=theta, dense=dense
    ).to(device)

    for epoch in range(epochs):
        model.train(mode=True)
        print(f"epoch: {epoch}")
        train_data_iterator = tqdm(train_data_loader)
        with Interrupter():
            for data, label in train_data_iterator:
                input_spikes = data.reshape(-1, 2, x_max * y_max, t_max)
                output_spikes = model.forward(
                    input_spikes, label.to(device), bias_decay=decay)
                # output_spikes: bacth, channel, neuro, time
                accurate = (output_spikes.sum((-3, -2, -1)) > 0).logical_and(
                    output_spikes.sum((-2, -1)).argmax(-1) == label.to(device)).sum()
                train_data_iterator.set_description(
                    f'{model.describe()}; {output_spikes.sum().int()}, {accurate}')

        model.train(mode=False)
        torch.save(model.state_dict(), f'{model_path}_epoch{epoch}')

        tracer = SpikesTracer(10)
        with Interrupter():
            test_data_iterator = tqdm(test_data_loader)
            for data, label in test_data_iterator:
                input_spikes = data.reshape(-1, 2, x_max * y_max, t_max)
                output_spikes = model.forward(input_spikes)

                y_preds = tracer.get_predict(output_spikes)
                tracer.add_sample(label.numpy(), y_preds)

                test_data_iterator.set_description(
                    '; '.join(
                        f'{k}: {v}'
                        for k, v in tracer.describe_batch_spikes(output_spikes).items()
                    )
                )
        tracer.describe()

    return 0


if __name__ == '__main__':
    exit(main())
