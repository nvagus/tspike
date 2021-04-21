import click
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import NMnistSampled
from tnn import AutoMatchingMatrix, FullDualColumn


class Interrupter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            return True
        return exc_type is None


@click.command()
@click.option('-g', '--gpu', default=0)
@click.option('-e', '--epochs', default=1)
@click.option('-b', '--batch', default=32)
@click.option('-x', '--x-max', default=34)
@click.option('-y', '--y-max', default=34)
@click.option('-t', '--t-max', default=256)
@click.option('-f', '--forced-dep', default=0)
@click.option('-d', '--dense', default=0.15)
@click.option('-w', '--w-init', default=0.5)
@click.option('-s', '--step', default=16)
@click.option('-l', '--leak', default=32)
@click.option('-k', '--dual', default=0.05)
@click.option('--capture', default=0.20)
@click.option('--backoff', default=-0.20)
@click.option('--search', default=0.001)
@click.option('-S/-U', '--supervised/--unsupervised', default=True)
@click.option('--train-path', default='data/n-mnist/TrainSP')
@click.option('--test-path', default='data/n-mnist/TestSP')
@click.option('--model-path', default='model/n-mnist-du')
def main(
    gpu, batch, epochs, supervised,
    x_max, y_max, t_max,
    step, leak,
    forced_dep, dense, w_init, dual,
    capture, backoff, search,
    train_path, test_path, model_path,
    **kwargs
):
    if torch.cuda.is_available():
        dev = f'cuda:{gpu}'
    else:
        dev = 'cpu'
    device = torch.device(dev)

    print(
        f'Device: {device}, Batch: {batch}, Epochs: {epochs}, Supervised: {supervised}')
    print(f'Forced Dep: {forced_dep}, Dense: {dense}, Weight Init: {w_init}')

    train_data_loader = DataLoader(NMnistSampled(
        train_path, x_max, y_max, t_max, device=device), shuffle=True, batch_size=batch)
    test_data_loader = DataLoader(NMnistSampled(
        test_path, x_max, y_max, t_max, device=device), batch_size=batch)

    model = FullDualColumn(
        x_max * y_max, 1, input_channel=2, output_channel=10,
        step=step, leak=leak,
        dense=dense, fodep=forced_dep, w_init=w_init, dual=dual
    ).to(device)

    def descriptor():
        return (
            f"{','.join('{:.0f}'.format(x) for x in model.weight_pos.sum(axis=1))}; "
            f"{','.join('{:.0f}'.format(x) for x in model.weight_neg.sum(axis=1))}"
        )

    for epoch in range(epochs):
        model.train(mode=True)
        print(f"epoch: {epoch}")
        train_data_iterator = tqdm(train_data_loader)
        train_data_iterator.set_description(descriptor())
        with Interrupter():
            for data, label in train_data_iterator:
                input_spikes = data.reshape(-1, 2, x_max * y_max, t_max)
                if supervised:
                    output_spikes = model.forward(
                        input_spikes, label.to(device), mu_capture=capture, mu_backoff=backoff, mu_search=search)
                else:
                    output_spikes = model.forward(input_spikes)
                # output_spikes: bacth, channel, neuro, time
                accurate = (output_spikes.sum((-3, -2, -1)) > 0).logical_and(
                    output_spikes.sum((-2, -1)).argmax(-1) == label.to(device)).sum()
                train_data_iterator.set_description(
                    f'{descriptor()}; {output_spikes.sum()}, {accurate}')

        model.train(mode=False)
        auto_matcher = AutoMatchingMatrix(10, 10)
        with Interrupter():
            for data, label in tqdm(test_data_loader):
                input_spikes = data.reshape(-1, 2, x_max * y_max, t_max)
                output_spikes = model.forward(input_spikes)

                has_spikes = output_spikes.sum((-3, -2, -1)) > 0
                y_preds = output_spikes.sum((-2, -1)).argmax(-1)

                for has_spike, y_pred, y_true in zip(has_spikes.cpu().numpy(), y_preds.cpu().numpy(), label.numpy()):
                    if has_spike:
                        auto_matcher.add_sample(y_true, y_pred)

        with Interrupter():
            print(auto_matcher.mat)
            print(f'Coverage: {auto_matcher.mat.sum() / len(test_data_loader.dataset)}')
            auto_matcher.describe_print_clear()
            torch.save(model.state_dict(), model_path)

    return 0


if __name__ == '__main__':
    exit(main())
