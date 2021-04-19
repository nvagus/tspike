import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import NMnistSampled
from tnn import AutoMatchingMatrix


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
    
    def forward(self, input_data):
        output = self.layer(input_data)
        logits = torch.log_softmax(output, dim=1)
        return logits


class ConvModel(nn.Module):
    def __init__(self, input_channels, output_channels, input_size, output_size, kernel_size=3, stride=2):
        super(ConvModel, self).__init__()
        self.conv_layer = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride)
        self.lienar_layer = nn.Linear(
            output_channels * 
            int(np.floor((input_size[0] - kernel_size) / stride + 1)) * 
            int(np.floor((input_size[1] - kernel_size) / stride + 1)), 
            output_size)
    
    def forward(self, input_data):
        conv_output = self.conv_layer(input_data)
        linear_output = self.lienar_layer(nn.functional.relu(conv_output).flatten(start_dim=1))
        logits = torch.log_softmax(linear_output, dim=1)
        return logits


class ConvGRUModel(nn.Module):
    def __init__(self, input_channels, output_channels, input_size, output_size, kernel_size=3, stride=2):
        super(ConvGRUModel, self).__init__()
        self.conv_layer = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride)
        self.gru_layer = nn.GRU(
            output_channels * 
            int(np.floor((input_size[0] - kernel_size) / stride + 1)) * 
            int(np.floor((input_size[1] - kernel_size) / stride + 1)),
            output_size)
    
    def forward(self, input_data):
        batch, channel, x, y, t = input_data.shape
        input_data = input_data.permute(4, 0, 1, 2, 3).reshape(-1, channel, x, y)
        conv_output = self.conv_layer(input_data).reshape(t, batch, -1)
        gru_output, _ = self.gru_layer(
            nn.functional.relu(conv_output), 
            torch.zeros(1, batch, self.gru_layer.hidden_size).to(conv_output.device))
        logits = torch.log_softmax(gru_output[-1], dim=1)
        return logits


@click.command()
@click.option('-g', '--gpu', default=0)
@click.option('-e', '--epochs', default=10)
@click.option('-b', '--batch', default=1)
@click.option('-x', '--x-max', default=34)
@click.option('-y', '--y-max', default=34)
@click.option('-t', '--t-max', default=256)
@click.option('--train-path', default='data/n-mnist/TrainSP')
@click.option('--test-path', default='data/n-mnist/TestSP')
@click.option('--model-path', default='model/n-mnist-1')
@click.option('-m', '--model', default='linear')
def main(
    gpu, batch, epochs,
    x_max, y_max, t_max,
    train_path, test_path, model_path,
    model,
    **kwargs
):
    if torch.cuda.is_available():  
        dev = f'cuda:{gpu}' 
    else:  
        dev = 'cpu'
    device = torch.device(dev)
    
    train_data_loader = DataLoader(NMnistSampled(train_path, x_max, y_max, t_max), shuffle=True, batch_size=batch)
    test_data_loader = DataLoader(NMnistSampled(test_path, x_max, y_max, t_max), batch_size=batch)

    if model == 'linear':
        model = LinearModel(x_max * y_max, 10).to(device)
        def transform(data):
            batch = data.shape[0]
            return data.reshape(batch, 2, x_max * y_max, t_max).to(device).sum(axis=(-1, -3)).float()
    elif model == 'linear_t':
        model = LinearModel(2 * x_max * y_max * t_max, 10).to(device)
        def transform(data):
            batch = data.shape[0]
            return data.reshape(batch, 2 * x_max * y_max * t_max).to(device).float()
    elif model == 'conv':
        model = ConvModel(2, 16, (x_max, y_max), 10).to(device)
        def transform(data):
            batch = data.shape[0]
            return data.reshape(batch, 2, x_max, y_max, t_max).to(device).sum(axis=-1).float()
    elif model == 'conv_gru_t':
        model = ConvGRUModel(2, 16, (x_max, y_max), 10).to(device)
        def transform(data):
            batch = data.shape[0]
            return data.reshape(batch, 2, x_max, y_max, t_max).to(device).float()
    else:
        return 255

    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        training = tqdm(train_data_loader)
        for sample in training:
            data = transform(sample['data'])
            labels = sample['label'].to(device)
            optimizer.zero_grad()
            output = model.forward(data)
            loss = error(output, labels.cuda())
            training.set_description(f'loss={loss.detach().cpu().numpy():.4f}')
            loss.backward()
            optimizer.step()
            
        auto_matcher = AutoMatchingMatrix(10, 10)
        for sample in tqdm(test_data_loader):
            data = transform(sample['data'])
            labels = sample['label']
            output = model.forward(data).argmax(axis=-1).cpu()
            for y_pred, y_true in zip(output, labels):
                auto_matcher.add_sample(y_true, y_pred)
        
        print(auto_matcher.mat)
        auto_matcher.describe_print_clear()
        torch.save(model.state_dict(), model_path)

    return 0

if __name__ == '__main__':
    exit(main())
