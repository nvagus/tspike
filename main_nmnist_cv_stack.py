import os
import click
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataset import NMnistSampled
from tnn import ConvColumn, StackCV, SpikesTracer, FullDualColumn


class Interrupter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            return True
        return exc_type is None


def eval_callback(ctx, param, value):
    return eval(value)


@click.command()
@click.option('-g', '--gpu', default=0)
@click.option('-e', '--epochs', default=1)
@click.option('-b', '--batch', default=32)
@click.option('-x', '--x-max', default=34)
@click.option('-y', '--y-max', default=34)
@click.option('-t', '--t-max', default=256)
@click.option('-f', '--forced-dep', default=0)
@click.option('-d', '--dense', default='[0.3,0.01]', callback=eval_callback)
@click.option('-w', '--w-init', default=0.3)
@click.option('-s', '--step', default=4)
@click.option('-l', '--leak', default=8)
@click.option('-c', '--channel', default='[16,32]', callback=eval_callback)
@click.option('-p', '--pooling-kernel', default=4)
@click.option('--winners', default=0.5)
@click.option('--capture', default=0.2000)
@click.option('--backoff', default=-0.2000)
@click.option('--search', default=0.0005)
@click.option('--fc-capture', default=0.200)
@click.option('--fc-backoff', default=-0.200)
@click.option('--fc-search', default=0.001)
@click.option('--fc-neuron', default=1)
@click.option('--fc-winners', default=1)
@click.option('--fc-step', default=16)
@click.option('--fc-leak', default=32)
@click.option('--fc-dense', default=0.10)
@click.option('--fc-w-init', default=0.3)
@click.option('-r', '--depth-start', default=-1)
@click.option('--train-path', default='data/n-mnist/TrainSP')
@click.option('--test-path', default='data/n-mnist/TestSP')
@click.option('--model-path', default='model/n-mnist-cv-stack')
def main(
    gpu, batch, epochs,
    x_max, y_max, t_max,
    step, leak, winners,
    forced_dep, dense, w_init, channel, pooling_kernel,
    capture, backoff, search,
    fc_capture, fc_backoff, fc_search,
    fc_neuron, fc_winners,
    fc_step, fc_leak, fc_dense, fc_w_init,
    depth_start, train_path, test_path, model_path,
    **kwargs
):
    if torch.cuda.is_available():
        dev = f'cuda:{gpu}'
    else:
        dev = 'cpu'
    device = torch.device(dev)

    cv_stack_model_path = os.path.join(model_path, "cv")
    if not os.path.isdir(cv_stack_model_path):
        os.makedirs(cv_stack_model_path)

    print(
        f'Device: {device}, Batch: {batch}, Epochs: {epochs}')
    print(f'Forced Dep: {forced_dep}, Dense: {dense}, Weight Init: {w_init}')

    train_data_loader = DataLoader(NMnistSampled(
        train_path, x_max, y_max, t_max, device=device), shuffle=True, batch_size=batch)
    test_data_loader = DataLoader(NMnistSampled(
        test_path, x_max, y_max, t_max, device=device), batch_size=batch)

    model = StackCV(
        channels=[2] + channel,
        step=step, leak=leak, bias=0.5, winners=winners,
        pooling_kernel=pooling_kernel,
        fodep=forced_dep, w_init=w_init, dense=dense
    ).to(device)

    def descriptor(depth):
        return ','.join('{:.0f}'.format(x) for x in model.columns[depth].weight.sum((1, 2, 3)).detach())

    def tester_descriptor():
        max_print = 10
        s = f"{','.join(f'{x*100:.0f}' for x in tester.weight.mean(axis=1)[:max_print])}; "
        s += f"{','.join(f'{x*100:.0f}' for x in tester.bias[:max_print])}; "

        return s

    def othogonal(depth):
        weight = model.columns[depth].weight
        oc, ic, x, y = weight.shape
        w = weight.reshape(oc, -1)
        w = w / (w ** 2).sum(1, keepdim=True).sqrt()
        return (((w @ w.T) ** 2).mean() - 1 / oc).sqrt()

    if depth_start != -1:
        print("Starting from ", depth_start)
        depth_i_model_path = os.path.join(model_path, str(depth_start - 1))
        model.load_state_dict(torch.load(depth_i_model_path))
        print("Finished loading ", depth_i_model_path)

        # just to get the shape of output_spikes
        model.eval()
        for data, label in train_data_loader:
            input_spikes = data
            output_spikes = model.forward(
                input_spikes, depth_start - 1, mu_capture=capture, mu_backoff=backoff, mu_search=search)
            break

    else:
        depth_start = 0
        print("Fresh train from 0")

    for epoch in range(epochs):
        model.train(mode=True)
        for depth in range(depth_start, model.num_spikes):
            print(f"train epoch: {epoch}, depth: {depth}")
            train_data_iterator = tqdm(train_data_loader)
            with Interrupter():
                for data, label in train_data_iterator:

                    input_spikes = data
                    output_spikes = model.forward(
                        input_spikes, depth, mu_capture=capture, mu_backoff=backoff, mu_search=search)

                    train_data_iterator.set_description(
                        f'weight sum:{descriptor(depth)}; '
                        f'weight othogonal:{othogonal(depth):.4f}; '
                        f'total spikes:{output_spikes.sum().int()}; '
                        f'time coverage:{(output_spikes.sum((1, 2, 3)) > 0).float().mean() * 100:.2f}')

            depth_i_model_path = os.path.join(model_path, str(depth))
            print("saving", depth_i_model_path)
            torch.save(model.state_dict(), depth_i_model_path)

    spikes_tracer = SpikesTracer()
    model.train(mode=False)

    # build tester
    batch, channel, synapses_x, synapses_y, time = output_spikes.shape
    fc_fodep = time + fc_step + fc_leak
    tester = FullDualColumn(synapses_x * synapses_y, fc_neuron,
                            input_channel=channel, output_channel=10,
                            step=fc_step, leak=fc_leak, winners=fc_winners,
                            fodep=fc_fodep, w_init=fc_w_init, dense=fc_dense
                            ).to(device)

    for epoch in range(epochs):
        print(f"tester epoch: {epoch}")
        tester.train(mode=True)
        with Interrupter():
            train_data_iterator = tqdm(train_data_loader)
            for data, label in train_data_iterator:
                output_spikes = model.forward(data)
                output_spikes = output_spikes.reshape(
                    -1, channel, synapses_x * synapses_y, time)
                output_spikes = tester.forward(
                    output_spikes, labels=label.to(device),
                    mu_capture=fc_capture, mu_backoff=fc_backoff, mu_search=fc_search)

                y_preds = output_spikes.sum((-2, -1)).argmax(-1)
                accurate = (output_spikes.sum((-3, -2, -1)) > 0).logical_and(
                    output_spikes.sum((-2, -1)).argmax(-1) == label.to(device)).sum()
                train_data_iterator.set_description(
                    f'{tester_descriptor()}; {output_spikes.sum()}, {accurate}')

        # make prediction
        tester.train(mode=False)
        tester_model_path = os.path.join(model_path, "fc")
        torch.save(model.state_dict(), tester_model_path)

        with Interrupter():
            for data, label in tqdm(test_data_loader):
                output_spikes = model.forward(data)
                output_spikes = output_spikes.reshape(
                    -1, channel, synapses_x * synapses_y, time)

                output_spikes = tester.forward(
                    output_spikes, labels=label.to(device))

                y_preds = spikes_tracer.get_predict(output_spikes)
                spikes_tracer.add_sample(label.numpy(), y_preds)

        spikes_tracer.describe_print_clear()

    return 0


if __name__ == '__main__':
    exit(main())
