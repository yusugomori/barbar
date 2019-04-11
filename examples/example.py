import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from barbar import Bar


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 200)
        self.l2 = nn.Linear(200, 10)

    def forward(self, x):
        h = self.l1(x)
        h = torch.relu(h)
        y = self.l2(h)
        return y


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Load data
    '''
    root = os.path.join('~', '.torch', 'mnist')
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda x: x.view(-1),
                                    lambda x: x / 255.])
    mnist_train = datasets.MNIST(root=root,
                                 download=True,
                                 train=True,
                                 transform=transform)
    train_dataloader = DataLoader(mnist_train,
                                  batch_size=100,
                                  shuffle=True)

    '''
    Build model
    '''
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers.Adam(model.parameters())

    '''
    Train model
    '''
    def train_step(x, t):
        model.train()
        preds = model(x)
        loss = criterion(preds, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    epochs = 30

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch+1))
        train_loss = 0.
        train_acc = 0.

        for idx, (x, t) in enumerate(Bar(train_dataloader)):
            x, t = x.to(device), t.to(device)
            loss, preds = train_step(x, t)
            train_loss += loss.item()
            train_acc += \
                accuracy_score(t.tolist(),
                               preds.argmax(dim=-1).tolist())

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        print('loss: {:.3}, acc: {:.3f}'.format(
            train_loss,
            train_acc
        ))
