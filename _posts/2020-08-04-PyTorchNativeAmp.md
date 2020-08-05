---
comments: true
author: krishan
layout: post
categories: deeplearning
title: Using PyTorch 1.6 native AMP
description: Showcasing how to use native amp's autocast() and Gradscaler through simple example model.
---

This tutorial provides step by step instruction for using native amp introcuded in PyTorch 1.6. Often times, its good to try stuffs using simple examples especially if they are related to graident updates. Scientists need to be careful while using mixed precission and write proper test cases. A single misstep can result is model divergence or unexpected error. This tutorial uses a simple 1x1 linear layer and converts and FP32 model training to mixed precission model training. Weights and Gradients are printed at every stage to ensure correctness. 

[PyTorch official documentation](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)


```python
import torch
print('torch version' , torch.__version__)

!nvidia-smi
!cat /usr/local/cuda/version.txt
```

    torch version 1.6.0
    Wed Aug  5 02:29:28 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 450.36.06    Driver Version: 450.36.06    CUDA Version: 11.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla V100-PCIE...  On   | 00000001:00:00.0 Off |                    0 |
    | N/A   32C    P0    28W / 250W |      0MiB / 16160MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-PCIE...  On   | 00000002:00:00.0 Off |                    0 |
    | N/A   30C    P0    23W / 250W |      0MiB / 16160MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    CUDA Version 10.1.243


# Install pytorch

Open terminal

    conda env list
    conda activate azureml_py36_pytorch
    conda install pytorch=1.6 torchvision cudatoolkit=10.1 -c pytorch

# Create a dummy model


```python
torch.manual_seed(47)


class MyModel(torch.nn.Module):
    def __init__(self, input_size = 1):
        super().__init__()
        self.linear = torch.nn.Linear(input_size,1)
        
    def forward(self, x):
        return self.linear(x)

model = MyModel()
model
```




    MyModel(
      (linear): Linear(in_features=1, out_features=1, bias=True)
    )




```python
#print parameters
def print_model_params(model):
    for name, param in model.named_parameters():
        print( 'Param Name = {}, value = {}, gradient = {}'
              .format(name , param.data, param.grad))
        
print_model_params(model)
```

    Param Name = linear.weight, value = tensor([[-0.8939]]), gradient = None
    Param Name = linear.bias, value = tensor([-0.9002]), gradient = None



```python
#input
x = torch.randn(1,1)
x
```




    tensor([[-0.0591]])



# Train model FP32


```python

optimizer = torch.optim.SGD(model.parameters(), lr = 1)

def train_step(model, x):
    print('\nRunning forward pass, input = ',x)
    output = model(x)
    print('output = ', output)
    
    print('\nRunning backward pass')
    output.backward()
    print('\nAfter backward pass')
    print_model_params(model)
    
    print('\nAfter optimizer step')
    optimizer.step()
    print('\nAfter updating model weights')
    
    print_model_params(model)
    optimizer.zero_grad()
    
    print('\nAfter setting gradients to zero')  
    print_model_params(model)
    
```


```python
train_step(model, x)
```

    
    Running forward pass, input =  tensor([[-0.0591]])
    output =  tensor([[-0.8473]], grad_fn=<AddmmBackward>)
    
    Running backward pass
    
    After backward pass
    Param Name = linear.weight, value = tensor([[-0.8939]]), gradient = tensor([[-0.0591]])
    Param Name = linear.bias, value = tensor([-0.9002]), gradient = tensor([1.])
    
    After optimizer step
    
    After updating model weights
    Param Name = linear.weight, value = tensor([[-0.8348]]), gradient = tensor([[-0.0591]])
    Param Name = linear.bias, value = tensor([-1.9002]), gradient = tensor([1.])
    
    After setting gradients to zero
    Param Name = linear.weight, value = tensor([[-0.8348]]), gradient = tensor([[0.]])
    Param Name = linear.bias, value = tensor([-1.9002]), gradient = tensor([0.])


# Train model with AMP


```python
from torch.cuda.amp import autocast, GradScaler

#grad scaler only works on GPU
model = model.to('cuda:0')
x = x.to('cuda:0')

optimizer = torch.optim.SGD(model.parameters(), lr = 1)
scaler = GradScaler(init_scale=4096) 


def train_step_amp(model, x):
    with autocast():
        print('\nRunning forward pass, input = ',x)
        output = model(x)
        print('output = ', output)
    
    print('\nRunning backward pass')
    scaler.scale(output).backward()
    print('\nAfter backward pass')
    print_model_params(model)
    
#     scaler.unscale_(optimizer) #optional 
#     print('\nAfter Unscaling')
#     print_model_params(model)
    
    scaler.step(optimizer) # do not use optimizer step as it will step over inf and nan values too.
    print('\nAfter updating model weights')
    
    print_model_params(model)
    optimizer.zero_grad()
    
    print('\nAfter setting gradients to zero')  
    print_model_params(model)
    
    scaler.update()
    
```


```python
train_step_amp(model, x)
```

    
    Running forward pass, input =  tensor([[-0.0591]], device='cuda:0')
    output =  tensor([[-1.8506]], device='cuda:0', dtype=torch.float16,
           grad_fn=<AddmmBackward>)
    
    Running backward pass
    
    After backward pass
    Param Name = linear.weight, value = tensor([[-0.8348]], device='cuda:0'), gradient = tensor([[-242.2500]], device='cuda:0')
    Param Name = linear.bias, value = tensor([-1.9002], device='cuda:0'), gradient = tensor([4096.], device='cuda:0')
    
    After updating model weights
    Param Name = linear.weight, value = tensor([[-0.7756]], device='cuda:0'), gradient = tensor([[-0.0591]], device='cuda:0')
    Param Name = linear.bias, value = tensor([-2.9002], device='cuda:0'), gradient = tensor([1.], device='cuda:0')
    
    After setting gradients to zero
    Param Name = linear.weight, value = tensor([[-0.7756]], device='cuda:0'), gradient = tensor([[0.]], device='cuda:0')
    Param Name = linear.bias, value = tensor([-2.9002], device='cuda:0'), gradient = tensor([0.], device='cuda:0')


The gradients are scaled and unscaled properly. Also the forward pass and backward pass are run using mixed precission. Although there is no easy verification, the timing difference between both runs will confirm the mixed precission training. This will be showcased in future blogs. 

Native amp support makes it easy to do fast experimentation without using apex related dependencies. 
