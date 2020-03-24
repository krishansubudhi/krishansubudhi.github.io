---
comments: true
author: krishan
layout: post
categories: deeplearning
title: Using Tensorboard in Pytorch
description: Using tensorboard in pytorch. This example uses windoes for the system commands. Linux and Mac will need slight modification in the powershell commands
---
Reference : https://pytorch.org/docs/stable/tensorboard.html

Clear everything first 


```python
! powershell "echo 'checking for existing tensorboard processes'"
! powershell "ps | Where-Object {$_.ProcessName -eq 'tensorboard'}"

! powershell "ps | Where-Object {$_.ProcessName -eq 'tensorboard'}| %{kill $_}"

! powershell "echo 'cleaning tensorboard temp dir'"
! powershell "rm -Force -Recurse $env:TEMP\.tensorboard-info\*" 

! powershell "ps | Where-Object {$_.ProcessName -eq 'tensorboard'}"
! powershell "rm -Force -Recurse runs\*"
```

    checking for existing tensorboard processes
    cleaning tensorboard temp dir
    

# Create Summary writer


```python
from torch.utils.tensorboard import SummaryWriter
# Writer will output to ./runs/ directory by default
writer = SummaryWriter('runs/testing_tensorboard_pt')
```

# Logging model graph and images 


```python
import torch
import torchvision
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()


```
![graphs](/assets/tensorboard-pt/graphs.jpg)
# Logging scalars ang grouping them


```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

#writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(),n_iter)
    writer.add_scalar('Loss/test', np.random.random(),n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(),n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(),n_iter)
```
# Run multiple scalars at once


```python
import math
#writer = SummaryWriter(log_dir = 'runs/multi_scalar')

for i,theta in enumerate(np.linspace(0, math.pi *2, 1000)):
    stats = {'sine': math.sin(theta),
            'cosine':math.cos(theta),
            'tangent':math.tan(theta)}
    writer.add_scalars('trignometry', stats, i )
writer.close()
```

![scalars](/assets/tensorboard-pt/scalars.jpg)
# Add histogram


```python
#writer = SummaryWriter(log_dir = 'runs/histogram')
for epoch in range(1,5):
    for i in range(10):
        writer.add_histogram('random_normal/epoch_'+str(epoch), torch.randn(1,1000)+i, i)
        writer.add_histogram('random_uniform/epoch_'+str(epoch), torch.rand(1,1000)+i, i)
    writer.close()
```

![histogram](/assets/tensorboard-pt/histogram.jpg)
# Add image


```python
#writer = SummaryWriter(log_dir = 'runs/images')
for step in range (3):
    # create a 100x100 random image and normalize
    random_image = np.random.randint(10000,size = (1,10000)).reshape(100,100)
    random_image = random_image/ 10000
    
    writer.add_image('random_image',
                     random_image,
                     step, 
                     dataformats = 'HW')
writer.close()

```
![images](/assets/tensorboard-pt/images.jpg)

# Add Text


```python
#writer = SummaryWriter(log_dir = 'runs/text')
writer.add_text('lstm', 'This is an lstm step 0', 0)
writer.add_text('lstm', 'This is an rnn step 10', 10)
writer.close()

```

![text](/assets/tensorboard-pt/text.jpg)
# Add hyper parameter


```python

for i in range(3):
    exp = f'exp{i}'
    with SummaryWriter('runs/'+exp) as w:
        w.add_hparams({'lr': 0.1*i, 'bsize': i},
                          {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
```

![hyperparameter](/assets/tensorboard-pt/hparams.jpg)

```python
writer.flush()
```


```python
# Run in a new anaconda powershell

# ! pwd
# ! dir runs
# ! tensorboard --logdir="C:\Users\krkusuk\projects\tensorboard\runs"
```
