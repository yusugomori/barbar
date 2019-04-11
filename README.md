# Barbar
Progress bar for deep learning training iterations.

![screenshot](https://user-images.githubusercontent.com/770299/55931402-3bb76000-5c60-11e9-9686-f6ae23adcaf0.png)



## Quick glance

```python
from barbar import Bar
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

mnist_train = datasets.MNIST(root=root,
                             download=True,
                             train=True)
train_dataloader = DataLoader(mnist_train,
                              batch_size=100,
                              shuffle=True)

model = MLP().to(device)

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch+1))

    for idx, (x, t) in enumerate(Bar(train_dataloader)):
        x, t = x.to(device), t.to(device)
        loss, preds = train_step(x, t)
```

```
Epoch: 1
60000/60000: [===============================>] - ETA 0.0s
Epoch: 2
28100/60000: [==============>.................] - ETA 4.1s
```
