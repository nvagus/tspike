import click
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import NMnistSampled
from tnn import AutoMatchingMatrix, FullColumn


def fc_model(x_max, y_max, w_max, ltd, dense=None, theta=None):
    model = FullColumn(
        x_max * y_max, 10, input_channel=2, 
        w_max=w_max, ltd=ltd, dense=dense, theta=theta
    )
    return model


@click.command()
@click.option('-b', '--batch', default=1)
@click.option('-x', '--x-max', default=34)
@click.option('-y', '--y-max', default=34)
@click.option('-t', '--t-max', default=256)
@click.option('-w', '--w-max', default=16)
@click.option('--train-path', default='data/n-mnist/TrainSP')
@click.option('--test-path', default='data/n-mnist/TestSP')
def main(
    batch,
    x_max, y_max, t_max, w_max,
    train_path, test_path,
    **kwargs
):
    train_data_loader = DataLoader(NMnistSampled(train_path, x_max, y_max, t_max), shuffle=True, batch_size=batch)
    test_data_loader = DataLoader(NMnistSampled(test_path, x_max, y_max, t_max), batch_size=batch)

    model = fc_model(x_max, y_max, w_max, ltd=t_max, dense=20)
    auto_matcher = AutoMatchingMatrix(10, 10)

    for epoch in range(10):
        print(f"epoch: {epoch}")
        for sample in tqdm(train_data_loader):
            input_spikes = sample['data'].reshape(batch, 2, x_max * y_max, t_max)
            output_spikes = model.forward(input_spikes)
            model.stdp(input_spikes, output_spikes)
        
        for sample in tqdm(test_data_loader):
            input_spikes = sample['data'].reshape(batch, 2, x_max * y_max, t_max)
            output_spikes = model.forward(input_spikes)
            prediction = output_spikes.sum(axis=-1).argmax()
            auto_matcher.add_sample(sample['label'], prediction)
        
        print(auto_matcher.mat)
        auto_matcher.describe_print_clear()
        print(f'weight sum: {model.weight.sum()}')

    return 0

if __name__ == '__main__':
    exit(main())
