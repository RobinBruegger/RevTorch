# RevTorch
Framework for creating (partially) reversible neural networks with PyTorch

RevTorch is introduced and explained in our paper [A Partially Reversible U-Net for Memory-Efficient Volumetric Image Segmentation](https://arxiv.org/abs/1906.06148),
which was accepted for presentation at [MICCAI 2019](https://www.miccai2019.org/). 

If you find this code helpful in your research please cite the following paper:
```
@article{PartiallyRevUnet2019Bruegger,
         author={Br{\"u}gger, Robin and Baumgartner, Christian F.
         and Konukoglu, Ender},
         title={A Partially Reversible U-Net for Memory-Efficient Volumetric Image Segmentation},
         journal={arXiv:1906.06148},
         year={2019},
```

## Installation
Use pip to install RevTorch:
```sh
$ pip install revtorch
```
RevTorch requires PyTorch. However, PyTorch is not included in the dependencies since the required PyTorch version is dependent on your system. Please install PyTorch following the instructions on the [PyTorch website](https://pytorch.org/).

## Usage
This example shows how to use the RevTorch framework.
```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import revtorch as rv

def train():
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    net = PartiallyReversibleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #logging stuff
            running_loss += loss.item()
            LOG_INTERVAL = 200
            if i % LOG_INTERVAL == (LOG_INTERVAL-1):  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / LOG_INTERVAL))
                running_loss = 0.0

class PartiallyReversibleNet(nn.Module):
    def __init__(self):
        super(PartiallyReversibleNet, self).__init__()

        #initial non-reversible convolution to get to 32 channels
        self.conv1 = nn.Conv2d(3, 32, 3)

        #construct reversible sequencce with 4 reversible blocks
        blocks = []
        for i in range(4):

            #f and g must both be a nn.Module whos output has the same shape as its input
            f_func = nn.Sequential(nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1))
            g_func = nn.Sequential(nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1))

            #we construct a reversible block with our F and G functions
            blocks.append(rv.ReversibleBlock(f_func, g_func))

        #pack all reversible blocks into a reversible sequence
        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

        #non-reversible convolution to get to 10 channels (one for each label)
        self.conv2 = nn.Conv2d(32, 10, 3)

    def forward(self, x):
        x = self.conv1(x)

        #the reversible sequence can be used like any other nn.Module. Memory-saving backpropagation is used automatically
        x = self.sequence(x)

        x = self.conv2(F.relu(x))
        x = F.avg_pool2d(x, (x.shape[2], x.shape[3]))
        x = x.view(x.shape[0], x.shape[1])
        return x

if __name__ == "__main__":
    train()
```

## Python version
Tested with Python 3.6 and PyTorch 1.1.0. Should work with any version of Python 3.