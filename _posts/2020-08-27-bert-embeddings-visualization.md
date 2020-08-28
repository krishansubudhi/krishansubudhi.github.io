---
comments: true
author: krishan
layout: post
categories: deeplearning
title: Visualizing Bert Embeddings
description: Visualize bert word Embeddings, position embeddings and contextual embeddings using TensorBoard
---

Set up tensorboard for pytorch by following this [blog](https://krishansubudhi.github.io/deeplearning/2020/03/24/tensorboard-pytorch.html). 

# Bert Embedding Layer

Bert has 3 types of embeddings
1. Word Embeddings
2. Position embeddings
3. Token Type embeddings



We will extract Bert Base Embeddings using Huggingface Transformer library and visualize them in tensorboard.

Clear everything first 


```python
! powershell "echo 'checking for existing tensorboard processes'"
! powershell "ps | Where-Object {$_.ProcessName -eq 'tensorboard'}"

! powershell "ps | Where-Object {$_.ProcessName -eq 'tensorboard'}| %{kill $_}"

! powershell "rm -Force -Recurse runs\*"
```


### Create a summary writer


```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/testing_tensorboard_pt')
```

Now let's fetch the pretrained bert Embeddings.

```python
import transformers
model = transformers.BertModel.from_pretrained('bert-base-uncased')
```


### Word embeddings

```python
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
words = tokenizer.vocab.keys()
word_embedding = model.embeddings.word_embeddings.weight
writer.add_embedding(word_embedding,
                         metadata  = words,
                        tag = f'word embedding')
```
### Position Embeddings

```python
position_embedding = model.embeddings.position_embeddings.weight
writer.add_embedding(position_embedding,
                         metadata  = np.arange(position_embedding.shape[0]),
                        tag = f'position embedding')
```
### Token type Embeddings

```python
token_type_embedding = model.embeddings.token_type_embeddings.weight
writer.add_embedding(token_type,
                         metadata  = np.arange(token_type_embedding.shape[0]),
                        tag = f'tokentype embeddings')
```


```python
writer.close()
```



### Run tensorboard
From the same folder as the notebook
```powershell 
tensorboard --logdir="C:\Users\...<current notebook folder path>\runs"
```

### Visualizations

1. All the country names are closer to *India* embeddings.
    ![word_india](/assets/bert-embedding-vis/word_india.jpg)



2. All the social networking site names are closer to *Facebook* embeddings.
![word_facebook](/assets/bert-embedding-vis/word_facebook.jpg)
3. Embedding of numbers are closer to one another.
![word_numbers](/assets/bert-embedding-vis/word_numbers.jpg)

4. Unused embeddings are closer.
![word_unused](/assets/bert-embedding-vis/word_unused.jpg)

5. In UMAP visualization, positional embeddings from 1-128 are showing one distribution while 128-512 are showing different distribution. This is probably because bert is pretrained in two phases. Phase 1 has 128 sequence length and phase 2 had 512.
![pos_umap](/assets/bert-embedding-vis/pos_umap.jpg)

# Contextual Embeddings

The power of BERT lies in it's ability to change representation based on context.
Now let's take few examples and see if embeddings change based on context.

For this we will only take the embeddings for final layer as those have the maximum high level context.

Dataset with different word senses will be the best way to visualize the representations.I used this word sense disambiguation dataset from Kaggle for analysis.
https://www.kaggle.com/udayarajdhungana/test-data-for-word-sense-disambiguation

Download and unzip



```python
# !pip install xlrd
import pandas as pd
examples = pd.read_excel('test data for WSD evaluation _2905.xlsx')
```


```python
pd.set_option('display.max_colwidth', 1000)
examples = examples.set_index(examples.sn)
```


```python
examples[examples['polysemy_word']=='bank']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sn</th>
      <th>sentence/context</th>
      <th>polysemy_word</th>
    </tr>
    <tr>
      <th>sn</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>I have bank account.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Loan amount is approved by the bank.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>He returned to office after he deposited cash in the bank.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>They started using new software in their bank.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>he went to bank balance inquiry.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>I wonder why some bank have more interest rate than others.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>You have to deposit certain percentage of your salary in the bank.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>He took loan from a Bank.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>he is waking along the river bank.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>The red boat in the bank is already sold.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>Spending time on the bank of Kaligandaki river was his way of enjoying in his childhood.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>He was sitting on sea bank with his friend</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>She has always dreamed of spending a vacation on a bank of Caribbean sea.</td>
      <td>bank</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>Bank of a river is very pleasant place to enjoy.</td>
      <td>bank</td>
    </tr>
  </tbody>
</table>
</div>




```python
model.eval()
context_embeddings = []
labels = []
with torch.no_grad():
    for record in examples.to_dict('record'):
        ids = tokenizer.encode(record['sentence/context'])
        tokens = tokenizer.convert_ids_to_tokens(ids)
        #print(tokens)
        bert_output = model.forward(torch.tensor(ids).unsqueeze(0),encoder_hidden_states = True)
        final_layer_embeddings = bert_output[0][-1]
        #print(final_layer_embeddings)
        
        for i, token in enumerate(tokens):
            if record['polysemy_word'].lower().startswith(token.lower()):
                #print(f'{record["sn"]}_{token}', final_layer_embeddings[i])
                context_embeddings.append(final_layer_embeddings[i])
                labels.append(f'{record["sn"]}_{token}')
#         break
        
# print(context_embeddings, labels)
```


```python
writer.add_embedding(torch.stack(context_embeddings),
                         metadata  = labels,
                        tag = f'contextual embeddings')
```
```python
writer.close()
```
### Restart tensorboard.
Delete existing logs if necessary and create the writer again using the instructions on top. This will speed up the loading.
```
ps | Where-Object {$_.ProcessName -eq 'tensorboard'}| %{kill $_}
tensorboard --logdir="<current dir path>\runs"
```
Open tensorboard UI in browser. It might take a while to load the embeddings. Keep refreshing the browser.

[http://localhost:6006/#projector&run=testing_tensorboard_pt](http://localhost:6006/#projector&run=testing_tensorboard_pt)

### Visualize contextual embeddings
Now same words with different meanings should be farther apart. Let's analyze the word **bank**  which has 2 different meanings. example 1-8 refer to banks as *financial institutes*, while example 9-14 use **bank** mostly as *the land alongside or sloping down to a river or lake.*

Let's see if Bert was able to figure this out

### Banks as financial institutes
![bank_1](/assets/bert-embedding-vis/bank_1.jpg)

Embeddings of **bank** in examples 9-14 are not close to the **bank** embeddings in 9-14. They are close to **bank** embeddings in example 2-8.
### Banks as river sides
**bank** embedding of example 9 is closer to **bank** embeddings of example 10-14
![bank_9](/assets/bert-embedding-vis/bank_9.jpg)
