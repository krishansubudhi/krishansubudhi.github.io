---
author: krishan
layout: post
categories: deeplearning
title: Bert Memory Consumption
---

This document analyses the memory usage of Bert Base and Bert Large for different sequences.
Additionally, the document provides memory usage without grad and finds that gradients consume most of the GPU memory for one Bert forward pass.
This also analyses the maximum batch size that can be accomodated for both Bert base and large. All the tests were conducted in Azure NC24sv3 machines

CORE	|RAM	|STORAGE	| GPU
--- | --- | --- | ---
24	|448 GiB	|2,948 GiB	|4X V100


```python
import torch
torch.cuda.is_available(),torch.cuda.device_count()
```

    (True, 4)



### Python logging
https://stackoverflow.com/questions/1943747/python-logging-before-you-run-logging-basicconfig


```python
from pytorch_transformers import *
import sys,logging
logging.root.handlers = []
logging.basicConfig(level="INFO", format = '%(asctime)s:%(levelname)s: %(message)s' ,stream = sys.stdout)
logger = logging.getLogger(__name__)
logger.info('hello')
```

    2019-09-18 15:50:17,776:INFO: hello


# BERT memory usage


```python
def check_memory():
    logger.info('GPU memory: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))
```

## Forward pass : Bert Base 


```python
device = torch.device('cuda')
torch.cuda.empty_cache()
check_memory()
```

    2019-09-18 15:50:17,826:INFO: GPU memory: 0.0



```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
logger.info('moving model to GPU')
gpu_model = model.to(device)
check_memory()
```

    2019-09-18 15:50:18,065:INFO: loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json from cache at /home/krishan/.cache/torch/pytorch_transformers/4dad0251492946e18ac39290fcfe91b89d370fee250efe9521476438fe8ca185.bf3b9ea126d8c0001ee8a1e8b92229871d06d36d8808208cc2449280da87785c
    2019-09-18 15:50:18,066:INFO: Model config {
      "attention_probs_dropout_prob": 0.1,
      "finetuning_task": null,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "num_labels": 2,
      "output_attentions": false,
      "output_hidden_states": false,
      "torchscript": false,
      "type_vocab_size": 2,
      "vocab_size": 30522
    }
    
    2019-09-18 15:50:18,226:INFO: loading weights file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin from cache at /home/krishan/.cache/torch/pytorch_transformers/aa1ef1aede4482d0dbcd4d52baad8ae300e60902e88fcb0bebdec09afd232066.36ca03ab34a1a5d5fa7bc3d03d55c4fa650fed07220e2eeebc06ce58d0e9a157
    2019-09-18 15:50:21,612:INFO: Weights of BertForSequenceClassification not initialized from pretrained model: ['classifier.weight', 'classifier.bias']
    2019-09-18 15:50:21,615:INFO: Weights from pretrained model not used in BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
    2019-09-18 15:50:21,618:INFO: moving model to GPU
    2019-09-18 15:50:23,644:INFO: GPU memory: 418.0



```python
#Run classification for a batch of 64 tensors
def run_bert(x):
    check_memory()
    logger.info('moving tensors to GPU')
    x = x.to(device)
    check_memory()
    logger.info('Running bert forward on x')
    yhat = gpu_model(x)
    check_memory()
    logger.info(f'yhat[0].requires_grad = {yhat[0].requires_grad} . Detaching yhat')
    yhat = yhat[0].detach()
    logger.info(f'x shape = {x.shape}, yhat.shape = {yhat.shape}')
    check_memory()
```

### 512


```python
x = torch.randint(low =1000, high = 30000 , size = (1,512))
run_bert(x)
```

    2019-09-18 15:50:23,675:INFO: GPU memory: 418.0
    2019-09-18 15:50:23,676:INFO: moving tensors to GPU
    2019-09-18 15:50:23,677:INFO: GPU memory: 418.0
    2019-09-18 15:50:23,678:INFO: Running bert forward on x
    2019-09-18 15:50:23,899:INFO: GPU memory: 1238.0
    2019-09-18 15:50:23,900:INFO: yhat[0].requires_grad = True . Detaching yhat
    2019-09-18 15:50:23,901:INFO: x shape = torch.Size([1, 512]), yhat.shape = torch.Size([1, 2])
    2019-09-18 15:50:23,902:INFO: GPU memory: 418.0


1 batch of 512 consumes 1237 - 418 = 829 MB of memory

This proves that holding the gradients consumes all the memories. (Almost double the model size)

Torch grad : https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95

#### Eval mode 


```python
gpu_model.eval()
run_bert(x)
```

    2019-09-18 15:50:23,908:INFO: GPU memory: 418.0
    2019-09-18 15:50:23,909:INFO: moving tensors to GPU
    2019-09-18 15:50:23,915:INFO: GPU memory: 418.0
    2019-09-18 15:50:23,915:INFO: Running bert forward on x
    2019-09-18 15:50:23,940:INFO: GPU memory: 1237.0
    2019-09-18 15:50:23,940:INFO: yhat[0].requires_grad = True . Detaching yhat
    2019-09-18 15:50:23,941:INFO: x shape = torch.Size([1, 512]), yhat.shape = torch.Size([1, 2])
    2019-09-18 15:50:23,942:INFO: GPU memory: 418.0


Even in eval mode, the grads are calculated.

#### Pass X with no grad


```python
x.requires_grad=False
run_bert(x)
```

    2019-09-18 15:50:23,947:INFO: GPU memory: 418.0
    2019-09-18 15:50:23,948:INFO: moving tensors to GPU
    2019-09-18 15:50:23,995:INFO: GPU memory: 418.0
    2019-09-18 15:50:23,995:INFO: Running bert forward on x
    2019-09-18 15:50:24,020:INFO: GPU memory: 1237.0
    2019-09-18 15:50:24,020:INFO: yhat[0].requires_grad = True . Detaching yhat
    2019-09-18 15:50:24,021:INFO: x shape = torch.Size([1, 512]), yhat.shape = torch.Size([1, 2])
    2019-09-18 15:50:24,022:INFO: GPU memory: 418.0


Still y required grad. even though x does not.

### 256 


```python
x = torch.randint(low =1000, high = 30000 , size = (1,256))
run_bert(x)
```

    2019-09-18 15:50:24,027:INFO: GPU memory: 418.0
    2019-09-18 15:50:24,028:INFO: moving tensors to GPU
    2019-09-18 15:50:24,076:INFO: GPU memory: 418.0
    2019-09-18 15:50:24,077:INFO: Running bert forward on x
    2019-09-18 15:50:24,147:INFO: GPU memory: 745.0
    2019-09-18 15:50:24,147:INFO: yhat[0].requires_grad = True . Detaching yhat
    2019-09-18 15:50:24,148:INFO: x shape = torch.Size([1, 256]), yhat.shape = torch.Size([1, 2])
    2019-09-18 15:50:24,149:INFO: GPU memory: 418.0


1 batch of 256 consumes 745 - 418 = 330 MB of memory.
1 batch of 512 consumes 829 MB of memory.

Hence for inputs with smaller sequences, bert consumes very less memory.

Question : Exactly where is extra memory (more than double from 256) being consumed for 512 sequence length module? 


### 128


```python
x = torch.randint(low =1000, high = 30000 , size = (1,128))
run_bert(x)
```

    2019-09-18 15:50:24,154:INFO: GPU memory: 418.0
    2019-09-18 15:50:24,154:INFO: moving tensors to GPU
    2019-09-18 15:50:24,155:INFO: GPU memory: 418.0
    2019-09-18 15:50:24,156:INFO: Running bert forward on x
    2019-09-18 15:50:24,181:INFO: GPU memory: 572.0
    2019-09-18 15:50:24,182:INFO: yhat[0].requires_grad = True . Detaching yhat
    2019-09-18 15:50:24,183:INFO: x shape = torch.Size([1, 128]), yhat.shape = torch.Size([1, 2])
    2019-09-18 15:50:24,183:INFO: GPU memory: 418.0


## Check maximum batch size


```python
#Run classification for a batch of 64 tensors
def run_max_batch_analysis(gpu_model,  nograd = False, seq_len = 512):
    def run_bert(x):
        check_memory()
        x = x.to(device)
        yhat = gpu_model(x)
        logger.info(f'x shape = {x.shape}, yhat.shape = {yhat[0].shape}')
        check_memory()
        
    def run_bert_nograd(x):
        with torch.no_grad():
            check_memory()
            x = x.to(device)
            yhat = gpu_model(x)
            logger.info(f'x shape = {x.shape}, yhat.shape = {yhat[0].shape}')
            check_memory()


    for batch in (4,8,16,32,64,128,256,512):
        try:
            x = torch.randint(low =1000, high = 30000 , size = (batch,seq_len))
            run_bert_nograd(x) if nograd else run_bert(x)
            logger.info(f'batch size {batch} successful.')
        except Exception as e:
            print(f'exception {type(e)} : {e}')
            break

```

### Bert Base


```python
del gpu_model
torch.cuda.empty_cache()
check_memory()
```

    2019-09-18 15:50:24,324:INFO: GPU memory: 418.0



```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
gpu_model = model.to(device)
check_memory()
```

    2019-09-18 15:50:24,512:INFO: loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json from cache at /home/krishan/.cache/torch/pytorch_transformers/4dad0251492946e18ac39290fcfe91b89d370fee250efe9521476438fe8ca185.bf3b9ea126d8c0001ee8a1e8b92229871d06d36d8808208cc2449280da87785c
    2019-09-18 15:50:24,513:INFO: Model config {
      "attention_probs_dropout_prob": 0.1,
      "finetuning_task": null,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "num_labels": 2,
      "output_attentions": false,
      "output_hidden_states": false,
      "torchscript": false,
      "type_vocab_size": 2,
      "vocab_size": 30522
    }
    
    2019-09-18 15:50:24,684:INFO: loading weights file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin from cache at /home/krishan/.cache/torch/pytorch_transformers/aa1ef1aede4482d0dbcd4d52baad8ae300e60902e88fcb0bebdec09afd232066.36ca03ab34a1a5d5fa7bc3d03d55c4fa650fed07220e2eeebc06ce58d0e9a157
    2019-09-18 15:50:27,988:INFO: Weights of BertForSequenceClassification not initialized from pretrained model: ['classifier.weight', 'classifier.bias']
    2019-09-18 15:50:27,990:INFO: Weights from pretrained model not used in BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
    2019-09-18 15:50:28,083:INFO: GPU memory: 418.0



```python
run_max_batch_analysis(gpu_model)
```

    2019-09-18 15:50:28,089:INFO: GPU memory: 418.0
    2019-09-18 15:50:28,188:INFO: x shape = torch.Size([4, 512]), yhat.shape = torch.Size([4, 2])
    2019-09-18 15:50:28,188:INFO: GPU memory: 3610.0
    2019-09-18 15:50:28,189:INFO: batch size 4 successful.
    2019-09-18 15:50:28,191:INFO: GPU memory: 418.0
    2019-09-18 15:50:28,429:INFO: x shape = torch.Size([8, 512]), yhat.shape = torch.Size([8, 2])
    2019-09-18 15:50:28,429:INFO: GPU memory: 6803.0
    2019-09-18 15:50:28,430:INFO: batch size 8 successful.
    2019-09-18 15:50:28,431:INFO: GPU memory: 418.0
    exception <class 'RuntimeError'> : CUDA out of memory. Tried to allocate 96.00 MiB (GPU 0; 11.17 GiB total capacity; 10.44 GiB already allocated; 51.94 MiB free; 100.42 MiB cached)


#### 128 seq length


```python
run_max_batch_analysis(gpu_model,False,128)
```

    2019-09-18 15:50:30,279:INFO: GPU memory: 418.0
    2019-09-18 15:50:30,305:INFO: x shape = torch.Size([4, 128]), yhat.shape = torch.Size([4, 2])
    2019-09-18 15:50:30,306:INFO: GPU memory: 1009.0
    2019-09-18 15:50:30,307:INFO: batch size 4 successful.
    2019-09-18 15:50:30,308:INFO: GPU memory: 418.0
    2019-09-18 15:50:30,381:INFO: x shape = torch.Size([8, 128]), yhat.shape = torch.Size([8, 2])
    2019-09-18 15:50:30,382:INFO: GPU memory: 1582.0
    2019-09-18 15:50:30,382:INFO: batch size 8 successful.
    2019-09-18 15:50:30,384:INFO: GPU memory: 418.0
    2019-09-18 15:50:30,517:INFO: x shape = torch.Size([16, 128]), yhat.shape = torch.Size([16, 2])
    2019-09-18 15:50:30,517:INFO: GPU memory: 2747.0
    2019-09-18 15:50:30,518:INFO: batch size 16 successful.
    2019-09-18 15:50:30,519:INFO: GPU memory: 418.0
    2019-09-18 15:50:30,775:INFO: x shape = torch.Size([32, 128]), yhat.shape = torch.Size([32, 2])
    2019-09-18 15:50:30,776:INFO: GPU memory: 5075.0
    2019-09-18 15:50:30,777:INFO: batch size 32 successful.
    2019-09-18 15:50:30,778:INFO: GPU memory: 418.0
    2019-09-18 15:50:31,291:INFO: x shape = torch.Size([64, 128]), yhat.shape = torch.Size([64, 2])
    2019-09-18 15:50:31,292:INFO: GPU memory: 9731.0
    2019-09-18 15:50:31,292:INFO: batch size 64 successful.
    2019-09-18 15:50:31,293:INFO: GPU memory: 418.0
    exception <class 'RuntimeError'> : CUDA out of memory. Tried to allocate 192.00 MiB (GPU 0; 11.17 GiB total capacity; 10.35 GiB already allocated; 145.94 MiB free; 101.96 MiB cached)


For lesser sequence length, the memory usage decreases with same ratio for all batch sizes. For example,

**Batch size 4 : memory consumed in forward pass**

512 seq length = 3610-418 = 3192

128 seq length = 1009-418 = 591

3192/591 = 5.4

512/128 = 4

**Batch size 8 : memory consumed in forward pass**

512 seq length = 6803-418 = 6385

128 seq length = 1582 -418 = 1164

6384/1187 = 5.48

512/128 = 4

#### No grad


```python
run_max_batch_analysis(gpu_model, nograd = True)
```

    2019-09-18 15:50:33,832:INFO: GPU memory: 418.0
    2019-09-18 15:50:33,855:INFO: x shape = torch.Size([4, 512]), yhat.shape = torch.Size([4, 2])
    2019-09-18 15:50:33,855:INFO: GPU memory: 418.0
    2019-09-18 15:50:33,856:INFO: batch size 4 successful.
    2019-09-18 15:50:33,857:INFO: GPU memory: 418.0
    2019-09-18 15:50:34,171:INFO: x shape = torch.Size([8, 512]), yhat.shape = torch.Size([8, 2])
    2019-09-18 15:50:34,172:INFO: GPU memory: 418.0
    2019-09-18 15:50:34,173:INFO: batch size 8 successful.
    2019-09-18 15:50:34,173:INFO: GPU memory: 418.0
    2019-09-18 15:50:34,785:INFO: x shape = torch.Size([16, 512]), yhat.shape = torch.Size([16, 2])
    2019-09-18 15:50:34,785:INFO: GPU memory: 418.0
    2019-09-18 15:50:34,786:INFO: batch size 16 successful.
    2019-09-18 15:50:34,788:INFO: GPU memory: 418.0
    2019-09-18 15:50:37,030:INFO: x shape = torch.Size([32, 512]), yhat.shape = torch.Size([32, 2])
    2019-09-18 15:50:37,031:INFO: GPU memory: 418.0
    2019-09-18 15:50:37,032:INFO: batch size 32 successful.
    2019-09-18 15:50:37,033:INFO: GPU memory: 418.0
    2019-09-18 15:50:39,476:INFO: x shape = torch.Size([64, 512]), yhat.shape = torch.Size([64, 2])
    2019-09-18 15:50:39,476:INFO: GPU memory: 418.0
    2019-09-18 15:50:39,477:INFO: batch size 64 successful.
    2019-09-18 15:50:39,479:INFO: GPU memory: 418.0
    2019-09-18 15:50:44,532:INFO: x shape = torch.Size([128, 512]), yhat.shape = torch.Size([128, 2])
    2019-09-18 15:50:44,533:INFO: GPU memory: 419.0
    2019-09-18 15:50:44,534:INFO: batch size 128 successful.
    2019-09-18 15:50:44,537:INFO: GPU memory: 418.0
    2019-09-18 15:50:55,386:INFO: x shape = torch.Size([256, 512]), yhat.shape = torch.Size([256, 2])
    2019-09-18 15:50:55,388:INFO: GPU memory: 419.0
    2019-09-18 15:50:55,389:INFO: batch size 256 successful.
    2019-09-18 15:50:55,394:INFO: GPU memory: 418.0
    exception <class 'RuntimeError'> : CUDA out of memory. Tried to allocate 6.00 GiB (GPU 0; 11.17 GiB total capacity; 4.92 GiB already allocated; 4.88 GiB free; 816.27 MiB cached)


Bert base with grad can accomodate a batch size of 8 while with no grad it can accomodate a batch size of 256.

### Bert Large


```python
del gpu_model
torch.cuda.empty_cache()
check_memory()
```

    2019-09-18 15:51:16,666:INFO: GPU memory: 418.0



```python
model = BertForSequenceClassification.from_pretrained('bert-large-uncased')
gpu_model = model.to(device)
check_memory()
```

    2019-09-18 15:51:17,114:INFO: loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json from cache at /home/krishan/.cache/torch/pytorch_transformers/6dfaed860471b03ab5b9acb6153bea82b6632fb9bbe514d3fff050fe1319ee6d.4c88e2dec8f8b017f319f6db2b157fee632c0860d9422e4851bd0d6999f9ce38
    2019-09-18 15:51:17,116:INFO: Model config {
      "attention_probs_dropout_prob": 0.1,
      "finetuning_task": null,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 1024,
      "initializer_range": 0.02,
      "intermediate_size": 4096,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "num_attention_heads": 16,
      "num_hidden_layers": 24,
      "num_labels": 2,
      "output_attentions": false,
      "output_hidden_states": false,
      "torchscript": false,
      "type_vocab_size": 2,
      "vocab_size": 30522
    }
    
    2019-09-18 15:51:17,287:INFO: loading weights file https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin from cache at /home/krishan/.cache/torch/pytorch_transformers/54da47087cc86ce75324e4dc9bbb5f66c6e83a7c6bd23baea8b489acc8d09aa4.4d5343a4b979c4beeaadef17a0453d1bb183dd9b084f58b84c7cc781df343ae6
    2019-09-18 15:51:27,360:INFO: Weights of BertForSequenceClassification not initialized from pretrained model: ['classifier.weight', 'classifier.bias']
    2019-09-18 15:51:27,361:INFO: Weights from pretrained model not used in BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
    2019-09-18 15:51:27,655:INFO: GPU memory: 1279.0



```python
run_max_batch_analysis(gpu_model)
```

    2019-09-18 15:51:27,660:INFO: GPU memory: 1279.0
    2019-09-18 15:51:27,811:INFO: x shape = torch.Size([4, 512]), yhat.shape = torch.Size([4, 2])
    2019-09-18 15:51:27,812:INFO: GPU memory: 9759.0
    2019-09-18 15:51:27,813:INFO: batch size 4 successful.
    2019-09-18 15:51:27,814:INFO: GPU memory: 1279.0
    exception <class 'RuntimeError'> : CUDA out of memory. Tried to allocate 128.00 MiB (GPU 0; 11.17 GiB total capacity; 10.33 GiB already allocated; 127.94 MiB free; 140.22 MiB cached)


#### 128 seq length


```python
run_max_batch_analysis(gpu_model,False,128)
```

    2019-09-18 15:51:29,780:INFO: GPU memory: 1279.0
    2019-09-18 15:51:30,663:INFO: x shape = torch.Size([4, 128]), yhat.shape = torch.Size([4, 2])
    2019-09-18 15:51:30,663:INFO: GPU memory: 2823.0
    2019-09-18 15:51:30,665:INFO: batch size 4 successful.
    2019-09-18 15:51:30,666:INFO: GPU memory: 1279.0
    2019-09-18 15:51:30,947:INFO: x shape = torch.Size([8, 128]), yhat.shape = torch.Size([8, 2])
    2019-09-18 15:51:30,947:INFO: GPU memory: 4367.0
    2019-09-18 15:51:30,949:INFO: batch size 8 successful.
    2019-09-18 15:51:30,950:INFO: GPU memory: 1279.0
    2019-09-18 15:51:31,464:INFO: x shape = torch.Size([16, 128]), yhat.shape = torch.Size([16, 2])
    2019-09-18 15:51:31,465:INFO: GPU memory: 7455.0
    2019-09-18 15:51:31,466:INFO: batch size 16 successful.
    2019-09-18 15:51:31,468:INFO: GPU memory: 1279.0
    exception <class 'RuntimeError'> : CUDA out of memory. Tried to allocate 32.00 MiB (GPU 0; 11.17 GiB total capacity; 10.42 GiB already allocated; 31.94 MiB free; 140.06 MiB cached)


#### No grad


```python
run_max_batch_analysis(gpu_model,nograd = True)
```

    2019-09-18 15:51:32,303:INFO: GPU memory: 1279.0
    2019-09-18 15:51:33,611:INFO: x shape = torch.Size([4, 512]), yhat.shape = torch.Size([4, 2])
    2019-09-18 15:51:33,612:INFO: GPU memory: 1279.0
    2019-09-18 15:51:33,612:INFO: batch size 4 successful.
    2019-09-18 15:51:33,613:INFO: GPU memory: 1279.0
    2019-09-18 15:51:34,701:INFO: x shape = torch.Size([8, 512]), yhat.shape = torch.Size([8, 2])
    2019-09-18 15:51:34,702:INFO: GPU memory: 1279.0
    2019-09-18 15:51:34,703:INFO: batch size 8 successful.
    2019-09-18 15:51:34,704:INFO: GPU memory: 1279.0
    2019-09-18 15:51:37,921:INFO: x shape = torch.Size([16, 512]), yhat.shape = torch.Size([16, 2])
    2019-09-18 15:51:37,922:INFO: GPU memory: 1279.0
    2019-09-18 15:51:37,923:INFO: batch size 16 successful.
    2019-09-18 15:51:37,923:INFO: GPU memory: 1279.0
    2019-09-18 15:51:42,348:INFO: x shape = torch.Size([32, 512]), yhat.shape = torch.Size([32, 2])
    2019-09-18 15:51:42,349:INFO: GPU memory: 1279.0
    2019-09-18 15:51:42,349:INFO: batch size 32 successful.
    2019-09-18 15:51:42,351:INFO: GPU memory: 1279.0
    2019-09-18 15:51:51,284:INFO: x shape = torch.Size([64, 512]), yhat.shape = torch.Size([64, 2])
    2019-09-18 15:51:51,285:INFO: GPU memory: 1279.0
    2019-09-18 15:51:51,286:INFO: batch size 64 successful.
    2019-09-18 15:51:51,288:INFO: GPU memory: 1279.0
    2019-09-18 15:52:09,143:INFO: x shape = torch.Size([128, 512]), yhat.shape = torch.Size([128, 2])
    2019-09-18 15:52:09,144:INFO: GPU memory: 1279.0
    2019-09-18 15:52:09,145:INFO: batch size 128 successful.
    2019-09-18 15:52:09,148:INFO: GPU memory: 1279.0
    exception <class 'RuntimeError'> : CUDA out of memory. Tried to allocate 4.00 GiB (GPU 0; 11.17 GiB total capacity; 7.25 GiB already allocated; 1.33 GiB free; 2.01 GiB cached)


## BertForPretraining

### BertBase


```python
del gpu_model
torch.cuda.empty_cache()
check_memory()
```

    2019-09-18 15:52:38,787:INFO: GPU memory: 1279.0



```python
check_memory()
model = BertForPreTraining.from_pretrained('bert-base-uncased')
gpu_model = model.to(device)
check_memory()
```

    2019-09-18 15:52:38,805:INFO: GPU memory: 1279.0
    2019-09-18 15:52:38,991:INFO: loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json from cache at /home/krishan/.cache/torch/pytorch_transformers/4dad0251492946e18ac39290fcfe91b89d370fee250efe9521476438fe8ca185.bf3b9ea126d8c0001ee8a1e8b92229871d06d36d8808208cc2449280da87785c
    2019-09-18 15:52:38,992:INFO: Model config {
      "attention_probs_dropout_prob": 0.1,
      "finetuning_task": null,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "num_labels": 2,
      "output_attentions": false,
      "output_hidden_states": false,
      "torchscript": false,
      "type_vocab_size": 2,
      "vocab_size": 30522
    }
    
    2019-09-18 15:52:39,155:INFO: loading weights file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin from cache at /home/krishan/.cache/torch/pytorch_transformers/aa1ef1aede4482d0dbcd4d52baad8ae300e60902e88fcb0bebdec09afd232066.36ca03ab34a1a5d5fa7bc3d03d55c4fa650fed07220e2eeebc06ce58d0e9a157
    2019-09-18 15:52:42,864:INFO: GPU memory: 425.0



```python
run_max_batch_analysis(gpu_model)
```

    2019-09-18 15:52:42,870:INFO: GPU memory: 425.0
    2019-09-18 15:52:42,923:INFO: x shape = torch.Size([4, 512]), yhat.shape = torch.Size([4, 512, 30522])
    2019-09-18 15:52:42,924:INFO: GPU memory: 3906.0
    2019-09-18 15:52:42,925:INFO: batch size 4 successful.
    2019-09-18 15:52:42,926:INFO: GPU memory: 425.0
    2019-09-18 15:52:43,349:INFO: x shape = torch.Size([8, 512]), yhat.shape = torch.Size([8, 512, 30522])
    2019-09-18 15:52:43,350:INFO: GPU memory: 7370.0
    2019-09-18 15:52:43,351:INFO: batch size 8 successful.
    2019-09-18 15:52:43,352:INFO: GPU memory: 425.0
    exception <class 'RuntimeError'> : CUDA out of memory. Tried to allocate 96.00 MiB (GPU 0; 11.17 GiB total capacity; 10.45 GiB already allocated; 11.94 MiB free; 134.05 MiB cached)


### Bert Large


```python
del gpu_model
torch.cuda.empty_cache()
check_memory()
```

    2019-09-18 15:52:46,251:INFO: GPU memory: 425.0



```python

model = BertForPreTraining.from_pretrained('bert-large-uncased')
check_memory()
gpu_model = model.to(device)
check_memory()
```

    2019-09-18 15:52:46,479:INFO: loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json from cache at /home/krishan/.cache/torch/pytorch_transformers/6dfaed860471b03ab5b9acb6153bea82b6632fb9bbe514d3fff050fe1319ee6d.4c88e2dec8f8b017f319f6db2b157fee632c0860d9422e4851bd0d6999f9ce38
    2019-09-18 15:52:46,480:INFO: Model config {
      "attention_probs_dropout_prob": 0.1,
      "finetuning_task": null,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 1024,
      "initializer_range": 0.02,
      "intermediate_size": 4096,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "num_attention_heads": 16,
      "num_hidden_layers": 24,
      "num_labels": 2,
      "output_attentions": false,
      "output_hidden_states": false,
      "torchscript": false,
      "type_vocab_size": 2,
      "vocab_size": 30522
    }
    
    2019-09-18 15:52:46,644:INFO: loading weights file https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin from cache at /home/krishan/.cache/torch/pytorch_transformers/54da47087cc86ce75324e4dc9bbb5f66c6e83a7c6bd23baea8b489acc8d09aa4.4d5343a4b979c4beeaadef17a0453d1bb183dd9b084f58b84c7cc781df343ae6
    2019-09-18 15:52:57,190:INFO: GPU memory: 0.0
    2019-09-18 15:52:57,464:INFO: GPU memory: 1283.0



```python
run_max_batch_analysis(gpu_model)
```

    2019-09-18 15:52:57,470:INFO: GPU memory: 1283.0
    2019-09-18 15:52:57,672:INFO: x shape = torch.Size([4, 512]), yhat.shape = torch.Size([4, 512, 30522])
    2019-09-18 15:52:57,673:INFO: GPU memory: 10058.0
    2019-09-18 15:52:57,674:INFO: batch size 4 successful.
    2019-09-18 15:52:57,675:INFO: GPU memory: 1283.0
    exception <class 'RuntimeError'> : CUDA out of memory. Tried to allocate 128.00 MiB (GPU 0; 11.17 GiB total capacity; 10.46 GiB already allocated; 43.94 MiB free; 92.09 MiB cached)

