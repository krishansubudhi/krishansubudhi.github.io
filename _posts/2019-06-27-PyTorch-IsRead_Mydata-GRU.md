---
author: krishan
layout: post
categories: deeplearning
title: PyTorch IsRead Predictor on my email
---
This is a GRU based RNN classifier to predict the read probability of a user from his/her email data.


```python
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
```


```python
input_file = r'mydata_3days.tsv'
df = pd.read_csv(input_file, sep='\t', header = None)
df.columns = ['subject','isread','time']

df = df[df.subject.notna()]
print(len(df))
df.head(5)

print(df.groupby('isread').count())
df.loc[df.isread].head(2)

#60,20,20
train, validate = np.split(df.sample(frac=1), [int(.6*len(df))])
train['split'] = 'train'
validate['split'] = 'val'
#test['split'] = 'test'

df = pd.concat([train,validate])

print(df.groupby('split')['isread'].count())

print('\nTotal isreads\n')
print(df.groupby('split')['isread'].sum())
df.head()
```

    1218
            subject  time
    isread               
    False      1090  1090
    True        128   128
    split
    train    730
    val      488
    Name: isread, dtype: int64
    
    Total isreads
    
    split
    train    69.0
    val      59.0
    Name: isread, dtype: float64





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
      <th>subject</th>
      <th>isread</th>
      <th>time</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>694</th>
      <td>PR - Updating package InventoryEntities from v...</td>
      <td>False</td>
      <td>2019-06-24T16:30:11.0000000Z</td>
      <td>train</td>
    </tr>
    <tr>
      <th>588</th>
      <td>PR - Updating build version to v16.02.1397.000...</td>
      <td>False</td>
      <td>2019-06-24T14:01:54.0000000Z</td>
      <td>train</td>
    </tr>
    <tr>
      <th>156</th>
      <td>Your scheduled experiment submission ( 09bd68e...</td>
      <td>False</td>
      <td>2019-06-23T15:30:44.0000000Z</td>
      <td>train</td>
    </tr>
    <tr>
      <th>436</th>
      <td>RE: Azure Cognitive Service Form Recognizer</td>
      <td>False</td>
      <td>2019-06-24T05:34:50.0000000Z</td>
      <td>train</td>
    </tr>
    <tr>
      <th>287</th>
      <td>PR - Updating package TorusGriffinSecrets from...</td>
      <td>False</td>
      <td>2019-06-25T06:58:06.0000000Z</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>




```python
class Tokenizer():
    def __init__(self, token_to_index, maxlen = 10):
        self.token_to_index = token_to_index
        self.index_to_token = {index:token for token,index in token_to_index.items()}
        self.maxlen = maxlen
    @classmethod
    def tokenize(self, text):
        text = text.lower()
        tokens = text.split()
        return tokens
    
    def tokens_to_tensor(self,tokens):
        tensor = np.zeros((self.maxlen,),dtype=np.int64)
        for i in range(min(len(tokens),self.maxlen)):
            token = tokens[i]
            tensor[i] = self.token_to_index[token] if token in self.token_to_index else 1
        return tensor
    
    def text_to_tensor(self,text):
        tokens  = Tokenizer.tokenize(text)
        indexes = self.tokens_to_tensor(tokens)
        return indexes
    
    @classmethod
    def create_tokenizer_from_df(cls,df):
        token_to_index = {}
        token_to_index['<MASK>'] = 0
        token_to_index['<PAD>'] = 1
        lengths = []
        for subject in df.subject.values:
            if type(subject) is  str:
                tokens = cls.tokenize(subject)
                lengths.append(len(tokens))
                for token in tokens:
                    if token not in token_to_index:
                        token_to_index[token] = len(token_to_index)
        lengths.sort()
        max_len = lengths[int(len(lengths)*0.9)]
        print('maxlen = ',max_len)
        return cls(token_to_index,max_len)
```


```python
%time tokenizer = Tokenizer.create_tokenizer_from_df(df)
print('vocab len = ', len(tokenizer.token_to_index))

tokenizer.text_to_tensor('Hello , how are you ?')
```

    maxlen =  15
    CPU times: user 3.46 ms, sys: 0 ns, total: 3.46 ms
    Wall time: 3.24 ms
    vocab len =  2518





    array([  1,   1,  79, 325, 218,   1,   0,   0,   0,   0,   0,   0,   0,
             0,   0])



##  Dataset


```python
class EmailDataSets(Dataset):
    def __init__(self,df, tokenizer):
        super(EmailDataSets, self).__init__()
        self.df = df
        self.active_df = df[df['split']=='train']
        self.tokenizer = tokenizer
    def set_split(self, split):
        self.active_df = df[df['split']== split]
    def __getitem__(self, index):
        row =  self.active_df.iloc[index]
        subject = row['subject']
        subject_tensor = tokenizer.text_to_tensor(subject)
        label = np.array(1 if row['isread'] else 0)
        
        return subject_tensor,label
    
    def __len__(self):
        return len(self.active_df)
```


```python
dataset = EmailDataSets(df, tokenizer)
print(dataset[0])
dataloader = DataLoader(dataset,batch_size=10)
for x,y in dataloader:
    print('x = ',x.shape,'y = ', y.shape)
    break
```

    (array([ 2,  3,  4,  5,  6,  7,  8,  3,  9, 10, 11, 12, 13,  0,  0]), array(0))
    x =  torch.Size([10, 15]) y =  torch.Size([10])


## Classifier


```python
class IsReadClassifier_RNN(nn.Module):
    def __init__(self, vocab_len, seq_len, hidden_dim, embedding_dim, num_layers=1):
        super(IsReadClassifier_RNN,self).__init__()
        self.embed = nn.Embedding(vocab_len,embedding_dim)
        self.GRU = nn.GRU(input_size= embedding_dim, hidden_size= hidden_dim, batch_first = True, \
                          bidirectional = False, num_layers = num_layers)
        self.fc1 = nn.Linear(hidden_dim,1)
        
    def forward(self, x, apply_sigmoid = False):
        '''
        args
        x = shape(batch,seq_len)
        '''
        yhat = self.embed(x)
        _,yhat = self.GRU(yhat) #**h_n** of shape `(num_layers * num_directions, batch, hidden_size)`
        yhat = yhat.permute(1,0,2)
        yhat = yhat[:, 0, :]
        yhat = yhat.contiguous().view(yhat.shape[0], -1)
        #print('yhat GRU = ',yhat.shape)
        yhat = self.fc1(yhat)
        
        if apply_sigmoid:
            yhat = torch.sigmoid(yhat)
        return yhat.squeeze()
```


```python
vocab_len = len(tokenizer.token_to_index)
c = IsReadClassifier_RNN(vocab_len, tokenizer.maxlen, 16, 64, 2)
print(c)
c.to(device)
for x,y in dataloader:
    x = x.to(device)
    y = y.to(device)
    print (x.dtype)
    yhat = c.forward(x,True)
    print (yhat.dtype)
    print(yhat, y)
    break
```

    IsReadClassifier_RNN(
      (embed): Embedding(2518, 64)
      (GRU): GRU(64, 16, num_layers=2, batch_first=True)
      (fc1): Linear(in_features=16, out_features=1, bias=True)
    )
    torch.int64
    torch.float32
    tensor([0.5130, 0.5123, 0.5351, 0.5406, 0.5145, 0.5351, 0.5476, 0.5396, 0.5351,
            0.5395], device='cuda:0', grad_fn=<SqueezeBackward0>) tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')


## Check for GPU


```python
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
print(device)
```

    cuda:0



```python
def calculate_val_loss(model, dataset):
    model.eval()
    dataset.set_split('val')
    dataloader = DataLoader(dataset, batch_size=64)
    loss = torch.nn.BCEWithLogitsLoss()
    
    total ,correct ,losses = 0,0,[]
    for x,y in dataloader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            y = torch.tensor(y, dtype = torch.float32)
            y = y.to(device)
            losses.append(loss(yhat,y).item())
            yhat = torch.tensor([1.0 if p>0.5 else 0.0 for p in yhat],dtype = torch.float32, device = device)
            correct += torch.sum(y==yhat).item()
            total += len(y)
    loss_avg = sum(losses)/len(losses)
    dataset.set_split('train')
    model.train()
    return loss_avg, correct/total 
```

## Start training 


```python
%%time
model = IsReadClassifier_RNN(vocab_len, tokenizer.maxlen, 16, 64, 2)
optimizer = torch.optim.Adam( model.parameters(), lr=0.001,)
epochs = 30
losses = []
loader = DataLoader(dataset,batch_size=16,)

loss = torch.nn.BCEWithLogitsLoss()
model.to(device)
model.train()
for e in range(epochs):
    for x, y in loader:
        x = x.to(device)
        y = torch.tensor(y, dtype = torch.float32)
        y = y.to(device)
        optimizer.zero_grad()
        yhat = model(x)
        #print(yhat, y)
        output = loss(yhat,y)
        
        output.backward()
        optimizer.step()


        losses.append(output.item())
    vloss,vacc = calculate_val_loss(model,dataset)
    print('loss = {}, vloss = {}, vacc = {}'.format(losses[-1],vloss,vacc))
```

    /anaconda/envs/py36/lib/python3.6/site-packages/ipykernel/__main__.py:13: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).


    loss = 0.14965474605560303, vloss = 0.41250869259238243, vacc = 0.8770491803278688
    loss = 0.12384742498397827, vloss = 0.38864874467253685, vacc = 0.8790983606557377
    loss = 0.11042094230651855, vloss = 0.3658701665699482, vacc = 0.8790983606557377
    loss = 0.09510580450296402, vloss = 0.34096507355570793, vacc = 0.8790983606557377
    loss = 0.0777280330657959, vloss = 0.2963421009480953, vacc = 0.8790983606557377
    loss = 0.05483933165669441, vloss = 0.2588787507265806, vacc = 0.8790983606557377
    loss = 0.030554382130503654, vloss = 0.2376303467899561, vacc = 0.9262295081967213
    loss = 0.020488834008574486, vloss = 0.2503366619348526, vacc = 0.9139344262295082
    loss = 0.014973670244216919, vloss = 0.23572852555662394, vacc = 0.9344262295081968
    loss = 0.012415190227329731, vloss = 0.25379848945885897, vacc = 0.930327868852459
    loss = 0.009942229837179184, vloss = 0.2555901985615492, vacc = 0.9323770491803278
    loss = 0.00861263182014227, vloss = 0.26605359092354774, vacc = 0.9323770491803278
    loss = 0.007612465415149927, vloss = 0.28402069583535194, vacc = 0.930327868852459
    loss = 0.0064257122576236725, vloss = 0.28632022719830275, vacc = 0.930327868852459
    loss = 0.005462944973260164, vloss = 0.27554639894515276, vacc = 0.9364754098360656
    loss = 0.005146912299096584, vloss = 0.3095714058727026, vacc = 0.9282786885245902
    loss = 0.004429324995726347, vloss = 0.2790329046547413, vacc = 0.9405737704918032
    loss = 0.003884026547893882, vloss = 0.28687122091650963, vacc = 0.9385245901639344
    loss = 0.0036064968444406986, vloss = 0.3291930612176657, vacc = 0.9262295081967213
    loss = 0.003162527456879616, vloss = 0.30498973093926907, vacc = 0.9385245901639344
    loss = 0.002883601002395153, vloss = 0.3186510391533375, vacc = 0.9405737704918032
    loss = 0.0029135930817574263, vloss = 0.30434839613735676, vacc = 0.9385245901639344
    loss = 0.0032340672332793474, vloss = 0.3205806640908122, vacc = 0.9262295081967213
    loss = 0.0027146299835294485, vloss = 0.31844725366681814, vacc = 0.9364754098360656
    loss = 0.0024211949203163385, vloss = 0.3328682454302907, vacc = 0.9323770491803278
    loss = 0.00218278169631958, vloss = 0.34755814447999, vacc = 0.9282786885245902
    loss = 0.0019903674256056547, vloss = 0.36618472915142775, vacc = 0.9241803278688525
    loss = 0.00183000264223665, vloss = 0.3771845642477274, vacc = 0.9241803278688525
    loss = 0.0016959089552983642, vloss = 0.3852106425911188, vacc = 0.9241803278688525
    loss = 0.0015792440390214324, vloss = 0.391275723464787, vacc = 0.9241803278688525
    CPU times: user 15 s, sys: 256 ms, total: 15.3 s
    Wall time: 15.3 s



```python
def predict (model,text,tokenizer):
    model.to(torch.device('cpu')).eval()
    with torch.no_grad():
        x = tokenizer.text_to_tensor(text)
        x = torch.from_numpy(x).unsqueeze(0)
        yhat = model(x,True)
    return yhat.item()
predict (model, 'hi', dataset.tokenizer )

for index,row in df[df['split']=='val'].head(20).iterrows():
    print('{}\t{:f}\t{}'.format( row.isread,predict(model, row.subject,dataset.tokenizer), row.subject+' ...'))
```

    True	0.992817	Your video has finished processing - "PreTraining AML-BERT code on Matrix" ...
    False	0.001732	FS: Mega bloks table and building bag. ...
    False	0.002356	<WTS> Kid's bike (12inch wheel size) ...
    False	0.001922	Everlast 100lb. Heavy Bag ...
    True	0.993263	INFORM:  INC22717582  training was reset abruptly ...
    False	0.003949	Re: Kenmore Coldspot 58289891 refrigerator  ...
    False	0.002657	Google home ...
    True	0.990992	RE: Code for BERT large training? ...
    False	0.001544	RE: Double Double:  ML.NET and Auto ML ...
    False	0.001425	RE: [PROD] Sev 3: ID 129379890: [WpA] [PROD] Tenants Not Delivered ...
    False	0.990373	Desk - $40 OBO ...
    False	0.001392	PR - Updating package LockBoxClient from version 16.... - MARS 403278 (MPU Build Account) ...
    False	0.001342	Your scheduled experiment submission ( 6bf8dba3-09c1-42a3-8cf9-0ebf120b845e ) is skipped. ...
    False	0.001565	RE: FS: 6.5kW Diesel Generator, ~100 hours, needs new battery - $500 ...
    False	0.001382	PR - Pyspark script for Tenant Details - MARS 404211 (Sneha Saran) ...
    True	0.002787	6/24-6/25 full day workshop 6/27 OOF ...
    True	0.003230	Need training data? ...
    False	0.001369	PR - Log improvement - MARS 404645 (Raj Srivastava) ...
    False	0.001389	PR - Updating package InventoryEntities from version... - MARS 404508 (MPU Build Account) ...
    False	0.001522	PR - Read TLC batch object from config file and Anan... - MARS 400619 (Komal Gyanani) ...

