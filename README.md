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

Barbar works best with PyTorch DataLoader, but it also works with custom DataLoader. Minimal DataLoader example can be written as follows:

```python
class CustomDataLoader(object):
    def __init__(self, dataset,
                 batch_size=100,
                 shuffle=False,
                 random_state=None):
        self.dataset = list(zip(dataset[0], dataset[1]))
        self.batch_size = batch_size
        self.shuffle = shuffle
        if random_state is None:
            random_state = np.random.RandomState(1234)
        self.random_state = random_state
        self._idx = 0
        self._reset()

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()

        x, y = \
            zip(*self.dataset[self._idx:(self._idx + self.batch_size)])

        # x = torch.Tensor(x)
        # y = torch.LongTensor(y)

        self._idx += self.batch_size

        return x, y

    def _reset(self):
        if self.shuffle:
            self.dataset = shuffle(self.dataset,
                                   random_state=self.random_state)
        self._idx = 0

mnist = datasets.fetch_openml('mnist_784', version=1,)
x, y = mnist.data.astype(np.float32), mnist.target.astype(np.int32)
x = x / 255.
x_train = x[:60000]
y_train = y[:60000]

train_dataloader = CustomDataLoader((x_train, y_train),
                                    batch_size=100,
                                    shuffle=True)
```

## Installation

- **Install Barbar from PyPI (recommended):**

```sh
pip install barbar
```

- **Alternatively: install Barbar from the GitHub source:**

First, clone Barbar using `git`:

```sh
git clone https://github.com/yusugomori/barbar.git
```

 Then, `cd` to the Barbar folder and run the install command:
```sh
cd barbar
sudo python setup.py install
```
