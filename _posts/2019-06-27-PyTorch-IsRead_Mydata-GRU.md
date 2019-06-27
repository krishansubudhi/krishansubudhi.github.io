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
input_file = r'.\mydata_3days.tsv'
df = pd.read_csv(input_file, sep='\t', header = None)
df.columns = ['subject','isread','time']

df = df[df.subject.notna()]
print(len(df))
df.head(10)
```

    1218
    




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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Your scheduled experiment submission ( 09bd68e...</td>
      <td>False</td>
      <td>2019-06-23T00:00:41.0000000Z</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Your scheduled experiment submission ( 09bd68e...</td>
      <td>False</td>
      <td>2019-06-25T00:00:39.0000000Z</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Your scheduled experiment submission ( 1c7e50c...</td>
      <td>False</td>
      <td>2019-06-23T00:02:41.0000000Z</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Shared Accommodation Available in 1bd/1ba</td>
      <td>False</td>
      <td>2019-06-25T00:00:49.0000000Z</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Your scheduled experiment submission ( e789fba...</td>
      <td>False</td>
      <td>2019-06-23T00:02:47.0000000Z</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RE: WTS - Samsung S10e Unlocked 128GB Prism Wh...</td>
      <td>False</td>
      <td>2019-06-25T00:01:40.0000000Z</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Re: Education vertical: AI demos</td>
      <td>False</td>
      <td>2019-06-23T00:06:38.0000000Z</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rental: Townhome in Woodbridge Community in Re...</td>
      <td>False</td>
      <td>2019-06-23T00:12:06.0000000Z</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RE: WTS - Samsung S10e Unlocked 128GB Prism Wh...</td>
      <td>False</td>
      <td>2019-06-25T00:03:07.0000000Z</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Re: Azure ML Dataset size limits</td>
      <td>False</td>
      <td>2019-06-23T00:30:13.0000000Z</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('isread').count()
df.loc[df.isread].head(10)
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
      <th>subject</th>
      <th>isread</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>RE: PR - Adding JT nuget package to substrate ...</td>
      <td>True</td>
      <td>2019-06-25T00:16:04.0000000Z</td>
    </tr>
    <tr>
      <th>45</th>
      <td>PR - Adding JT nuget package to substrate - Ni...</td>
      <td>True</td>
      <td>2019-06-25T00:27:56.0000000Z</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Re: Snorkel: Our Scenarios and Technical Support</td>
      <td>True</td>
      <td>2019-06-23T03:22:54.0000000Z</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Re: Snorkel: Our Scenarios and Technical Support</td>
      <td>True</td>
      <td>2019-06-23T04:02:58.0000000Z</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Tentative: PreTraining AML-BERT code on Matrix</td>
      <td>True</td>
      <td>2019-06-25T01:45:26.0000000Z</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Accepted: PreTraining AML-BERT code on Matrix</td>
      <td>True</td>
      <td>2019-06-25T01:56:47.0000000Z</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Graduation is here. Give The Times now.</td>
      <td>True</td>
      <td>2019-06-23T10:01:39.0000000Z</td>
    </tr>
    <tr>
      <th>146</th>
      <td>When to use embedding layer vs feaad forward l...</td>
      <td>True</td>
      <td>2019-06-23T14:11:40.0000000Z</td>
    </tr>
    <tr>
      <th>148</th>
      <td>Industry updates - Jun 23, 2019</td>
      <td>True</td>
      <td>2019-06-23T14:17:16.0000000Z</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Next stop: your future</td>
      <td>True</td>
      <td>2019-06-23T16:21:21.0000000Z</td>
    </tr>
  </tbody>
</table>
</div>




```python
#60,20,20
train, validate = np.split(df.sample(frac=1), [int(.6*len(df))])
train['split'] = 'train'
validate['split'] = 'val'
#test['split'] = 'test'

df = pd.concat([train,validate])
df.head()
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
      <th>subject</th>
      <th>isread</th>
      <th>time</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>Next stop: your future</td>
      <td>True</td>
      <td>2019-06-23T16:21:21.0000000Z</td>
      <td>train</td>
    </tr>
    <tr>
      <th>231</th>
      <td>ACTION…,ROWID…,DISTRIBUTION…,</td>
      <td>False</td>
      <td>2019-06-25T05:47:29.0000000Z</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1100</th>
      <td>PR - Adding sandbox type to sandbox entity - M...</td>
      <td>False</td>
      <td>2019-06-24T21:11:50.0000000Z</td>
      <td>train</td>
    </tr>
    <tr>
      <th>809</th>
      <td>RE: experiment fails immediately without node ...</td>
      <td>False</td>
      <td>2019-06-25T18:11:51.0000000Z</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1231</th>
      <td>Re: [PROD] Sev 3: ID 129379890: [WpA] [PROD] T...</td>
      <td>False</td>
      <td>2019-06-24T23:27:07.0000000Z</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.groupby('split')['isread'].count())
df.groupby('split')['isread'].sum()
```

    split
    train    730
    val      488
    Name: isread, dtype: int64
    




    split
    train    79.0
    val      49.0
    Name: isread, dtype: float64




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
len(tokenizer.token_to_index)
```

    maxlen =  15
    Wall time: 13 ms
    




    2518




```python
tokenizer.text_to_tensor('Hello , how are you ?')
```




    array([  1,   1, 200, 437, 607,   1,   0,   0,   0,   0,   0,   0,   0,
             0,   0], dtype=int64)




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
dataset[6]
```




    (array([41, 42, 43, 44, 45,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           dtype=int64), array(0))




```python
dataloader = DataLoader(dataset,batch_size=10)
for x,y in dataloader:
    print(x,y)
    break
```

    tensor([[ 2,  3,  4,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 7,  8,  9, 10, 11, 12, 10, 13,  8, 14, 15, 16, 17,  0,  0],
            [18, 19, 20, 21, 22, 23, 24, 25,  0,  0,  0,  0,  0,  0,  0],
            [18, 26, 27, 28, 29, 30, 31, 26, 32, 33, 34,  0,  0,  0,  0],
            [18, 35, 36, 37, 38, 39, 40,  0,  0,  0,  0,  0,  0,  0,  0],
            [41, 42, 43, 44, 45,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [18, 46, 47,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 4, 48, 19, 49, 50, 51, 52, 53, 54,  0,  0,  0,  0,  0,  0],
            [55, 56, 57, 58, 59, 60, 56, 61, 62, 38, 63, 64, 12, 65, 66]]) tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.int32)
    


```python
class IsReadClassifier_RNN(nn.Module):
    def __init__(self, vocab_len, seq_len, hidden_dim, embedding_dim):
        super(IsReadClassifier_RNN,self).__init__()
        self.embed = nn.Embedding(vocab_len,embedding_dim)
        self.GRU = nn.GRU(input_size= embedding_dim, hidden_size= hidden_dim, batch_first = True, \
                          bidirectional = False, num_layers = 1)
        self.fc1 = nn.Linear(hidden_dim,1)
        
    def forward(self, x, apply_sigmoid = False):
        '''
        args
        x = shape(batch,seq_len)
        '''
        yhat = self.embed(x)
        _,yhat = self.GRU(yhat) #**h_n** of shape `(num_layers * num_directions, batch, hidden_size)`
        yhat = yhat.squeeze()
        yhat = self.fc1(yhat)
        
        if apply_sigmoid:
            yhat = torch.sigmoid(yhat)
        return yhat.squeeze()
```


```python
vocab_len = len(tokenizer.token_to_index)
c = IsReadClassifier_RNN(vocab_len, tokenizer.maxlen, 16, 64)
print(c)
for x,y in dataloader:
    print (x.dtype)
    yhat = c.forward(x,True)
    print (yhat.dtype)
    print(yhat, y)
    break
```

    IsReadClassifier_RNN(
      (embed): Embedding(2518, 64)
      (GRU): GRU(64, 16, batch_first=True)
      (fc1): Linear(in_features=16, out_features=1, bias=True)
    )
    torch.int64
    torch.float32
    tensor([0.4254, 0.4149, 0.4632, 0.4279, 0.4396, 0.4291, 0.4216, 0.4082, 0.4090,
            0.5137], grad_fn=<SqueezeBackward0>) tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.int32)
    


```python
def calculate_val_loss(model, dataset):
    model.eval()
    dataset.set_split('val')
    dataloader = DataLoader(dataset, batch_size=64)
    loss = torch.nn.BCEWithLogitsLoss()
    
    total ,correct ,losses = 0,0,[]
    for x,y in dataloader:
        with torch.no_grad():
            yhat = model(x)
            y = torch.tensor(y, dtype = torch.float32)
            losses.append(loss(yhat,y).item())
            yhat = torch.tensor([1.0 if p>0.5 else 0.0 for p in yhat],dtype = torch.float32)
            correct += torch.sum(y==yhat).item()
            total += len(y)
    loss_avg = sum(losses)/len(losses)
    return loss_avg, correct/total 


```


```python
model = IsReadClassifier_RNN(vocab_len, tokenizer.maxlen, 16, 64)
optimizer = torch.optim.Adam(model.parameters())
epochs = 30
losses = []
loader = DataLoader(dataset,batch_size=16)

loss = torch.nn.BCEWithLogitsLoss()

model.train()
for e in range(epochs):
    for x, y in loader:
        y = torch.tensor(y, dtype = torch.float32)
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

    c:\users\krkusuk\appdata\local\continuum\anaconda3\envs\keras\lib\site-packages\ipykernel_launcher.py:12: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      if sys.path[0] == '':
    c:\users\krkusuk\appdata\local\continuum\anaconda3\envs\keras\lib\site-packages\ipykernel_launcher.py:11: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      # This is added back by InteractiveShellApp.init_path()
    

    loss = 0.3201567530632019, vloss = 0.42467159032821655, vacc = 0.8995901639344263
    loss = 0.21387353539466858, vloss = 0.3708634339272976, vacc = 0.8995901639344263
    loss = 0.17322340607643127, vloss = 0.35204739682376385, vacc = 0.8995901639344263
    loss = 0.15157471597194672, vloss = 0.33890034072101116, vacc = 0.8995901639344263
    loss = 0.13593831658363342, vloss = 0.32619384676218033, vacc = 0.8995901639344263
    loss = 0.12231801450252533, vloss = 0.3127310071140528, vacc = 0.8995901639344263
    loss = 0.11062949895858765, vloss = 0.29779060184955597, vacc = 0.8995901639344263
    loss = 0.10195116698741913, vloss = 0.27908417396247387, vacc = 0.8995901639344263
    loss = 0.09528098255395889, vloss = 0.25246208161115646, vacc = 0.8995901639344263
    loss = 0.08253885805606842, vloss = 0.22239124029874802, vacc = 0.9016393442622951
    loss = 0.0691109374165535, vloss = 0.18347833305597305, vacc = 0.9016393442622951
    loss = 0.055507831275463104, vloss = 0.14424133580178022, vacc = 0.9016393442622951
    loss = 0.044167518615722656, vloss = 0.1025908961892128, vacc = 0.9487704918032787
    loss = 0.033159881830215454, vloss = 0.07357885921373963, vacc = 0.9754098360655737
    loss = 0.02633027732372284, vloss = 0.05638555530458689, vacc = 0.9877049180327869
    loss = 0.0217863991856575, vloss = 0.04370216419920325, vacc = 0.9938524590163934
    loss = 0.01819724030792713, vloss = 0.033679374027997255, vacc = 0.9979508196721312
    loss = 0.01525272335857153, vloss = 0.026483369758352637, vacc = 1.0
    loss = 0.013103649951517582, vloss = 0.02231028047390282, vacc = 1.0
    loss = 0.011380737647414207, vloss = 0.01902481773868203, vacc = 1.0
    loss = 0.010042487643659115, vloss = 0.015868294634856284, vacc = 1.0
    loss = 0.008972586132586002, vloss = 0.013692040229216218, vacc = 1.0
    loss = 0.00808953307569027, vloss = 0.01224672282114625, vacc = 1.0
    loss = 0.007347091566771269, vloss = 0.011095929192379117, vacc = 1.0
    loss = 0.006714319344609976, vloss = 0.010128950816579163, vacc = 1.0
    loss = 0.00616874173283577, vloss = 0.009298252349253744, vacc = 1.0
    loss = 0.005693761631846428, vloss = 0.008574534673243761, vacc = 1.0
    loss = 0.005276733078062534, vloss = 0.007937004731502384, vacc = 1.0
    loss = 0.00490789907053113, vloss = 0.007367971760686487, vacc = 1.0
    loss = 0.004579600878059864, vloss = 0.006841991504188627, vacc = 1.0
    


```python
def predict (model,text,tokenizer):
    model.eval()
    with torch.no_grad():
        x = tokenizer.text_to_tensor(text)
        x = torch.from_numpy(x).unsqueeze(0)
        yhat = model(x,True)
    return yhat.item()
predict (model, 'hi', dataset.tokenizer )
```




    0.8477179408073425




```python
for index,row in df.head(20).iterrows():
    print('{}\t{:f}\t{}'.format( row.isread,predict(model, row.subject,dataset.tokenizer), row.subject[:10]+' ...'))
```

    True	0.498915	Next stop: ...
    False	0.658188	ACTION…,RO ...
    False	0.004343	PR - Addin ...
    False	0.004385	RE: experi ...
    False	0.004385	Re: [PROD] ...
    False	0.004874	RE: Double ...
    False	0.004863	FS: Colema ...
    False	0.011744	RE: PS4 Ga ...
    False	0.004351	Your sched ...
    True	0.899623	[IcM Surve ...
    False	0.004529	<Sell> Com ...
    False	0.004346	Your sched ...
    False	0.004358	PR - Updat ...
    False	0.004585	PR - Confi ...
    False	0.004451	RE: Ikea K ...
    False	0.004383	PR - Add S ...
    False	0.007211	<WTB> nest ...
    False	0.006934	FS: 3 Hone ...
    True	0.781654	Heads-up ...
    False	0.004346	Your sched ...
    
