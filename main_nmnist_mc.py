import click
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import NMnistSampled
from tnn import AutoMatchingMatrix, StackFullColumn


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
@click.option('-f', '--forced-dep', default='[0,256]', callback=eval_callback)
@click.option('-d', '--dense', default='[0.05,0.15]', callback=eval_callback)
@click.option('-a', '--theta', default='[30,50]', callback=eval_callback)
@click.option('-w', '--w-init', default='[0.5,0.5]', callback=eval_callback)
@click.option('-n', '--neurons', default='[32,10]', callback=eval_callback)
@click.option('-c', '--channels', default='[32,1]', callback=eval_callback)
@click.option('-s', '--step', default='[16,32]', callback=eval_callback)
@click.option('-l', '--leak', default='[32,64]', callback=eval_callback)
@click.option('-S/-U', '--supervised/--unsupervised', default=True)
@click.option('--train-path', default='data/n-mnist/TrainSP')
@click.option('--test-path', default='data/n-mnist/TestSP')
@click.option('--model-path', default='model/n-mnist-2')
def main(
    gpu, batch, epochs, supervised,
    x_max, y_max, t_max, 
    neurons, channels,
    step, leak,
    theta, dense, forced_dep, w_init,
    train_path, test_path, model_path,
    **kwargs
):
    if torch.cuda.is_available():  
        dev = f'cuda:{gpu}' 
    else:  
        dev = 'cpu'
    device = torch.device(dev)

    print(f'Device: {device}, Batch: {batch}, Epochs: {epochs}, Supervised: {supervised}')
    print(f'Forced Dep: {forced_dep}, Dense: {dense}, Weight Init: {w_init}')

    train_data_loader = DataLoader(NMnistSampled(train_path, x_max, y_max, t_max, device=device), shuffle=True, batch_size=batch)
    test_data_loader = DataLoader(NMnistSampled(test_path, x_max, y_max, t_max, device=device), batch_size=batch)

    model = StackFullColumn(
        [x_max * y_max, *neurons], [2, *channels], kernel=2,
        step=step, leak=leak,
        theta=theta, dense=dense, fodep=forced_dep, w_init=w_init
    ).to(device)
    
    def descriptor():
        return ','.join('{:.0f}'.format(c.weight.sum()) for c in model.columns)

    for epoch in range(epochs):
        model.train(mode=True)
        print(f"epoch: {epoch}")
        train_data_iterator = tqdm(train_data_loader)
        train_data_iterator.set_description(descriptor())
        with Interrupter():
            for data, label in train_data_iterator:
                input_spikes = data.reshape(-1, 2, x_max * y_max, t_max)
                if supervised:
                    output_spikes = model.forward(input_spikes, label.to(device))
                else:
                    output_spikes = model.forward(input_spikes)
                accurate = (output_spikes.sum((-3, -2, -1)) > 0).logical_and(output_spikes.sum((-3, -1)).argmax(-1) == label.to(device)).sum()
                train_data_iterator.set_description(f'{descriptor()}; {output_spikes.sum()}, {accurate}')
        
        model.train(mode=False)
        auto_matcher = AutoMatchingMatrix(10, 10)
        with Interrupter():
            for data, label in tqdm(test_data_loader):
                input_spikes = data.reshape(-1, 2, x_max * y_max, t_max)
                output_spikes = model.forward(input_spikes)

                has_spikes = output_spikes.sum((-3, -2, -1)) > 0
                y_preds = output_spikes.sum((-3, -1)).argmax(-1)

                for has_spike, y_pred, y_true in zip(has_spikes.cpu().numpy(), y_preds.cpu().numpy(), label.numpy()):
                    if has_spike:
                        auto_matcher.add_sample(y_true, y_pred)
        
        print(auto_matcher.mat)
        print(f'Coverage: {auto_matcher.mat.sum() / len(test_data_loader.dataset)}')
        auto_matcher.describe_print_clear()
        torch.save(model.state_dict(), model_path)

    return 0

if __name__ == '__main__':
    exit(main())
