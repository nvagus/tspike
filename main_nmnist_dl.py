import click
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from tnn import AutoMatchingMatrix

from dataset import NMnistSampled

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
    
    def forward(self, input_data):
        output = self.layer(input_data)
        logits = torch.log_softmax(output, dim=1)
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
            return data.reshape(batch, 2, x_max * y_max, t_max).to(device).sum(axis=(-1, -3)).float()
    elif model == 'linear_t':
        model = LinearModel(2 * x_max * y_max * t_max, 10).to(device)
        def transform(data):
            return data.reshape(batch, 2 * x_max * y_max * t_max).to(device).float()
    else:
        return 255

    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        for sample in tqdm(train_data_loader):
            data = transform(sample['data'])
            labels = sample['label'].to(device)
            optimizer.zero_grad()
            output = model.forward(data)
            loss = error(output, labels.cuda())
            loss.backward()
            optimizer.step()
            
        auto_matcher = AutoMatchingMatrix(10, 10)
        for sample in tqdm(test_data_loader):
            data = transform(sample['data'])
            labels = sample['label'].to(device)
            output = model.forward(data).argmax(axis=-1).cpu()
            for y_pred, y_true in zip(output, labels):
                auto_matcher.add_sample(y_true, y_pred)
        
        print(auto_matcher.mat)
        auto_matcher.describe_print_clear()
        torch.save(model.state_dict, model_path)

    return 0

if __name__ == '__main__':
    exit(main())
