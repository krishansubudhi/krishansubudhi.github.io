
A recurrent neural network (**RNN**) is a class of artificial neural network where connections between units form a directed cycle.

This is a complete example of an RNN multiclass classifier in pytorch. This uses a basic RNN cell and builds with minimal library dependency. 

[data file]:{{ site.url }}/download/surnames_split_krishan.csv

```python
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
```

### Create RNN layer using RNNCell


```python
class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first = False):
        '''
        Args:
        input_size (int): embedding size?
        batch_first (bool): whether the 0th dimension is batch
        '''
        super(ElmanRNN,self).__init__()
        
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        self.rnncell = nn.RNNCell(input_size,hidden_size)
    
    def get_initial_hidden_state(self, batch_size):
        '''
        Return all zeros
        '''
        return torch.zeros(batch_size,self.hidden_size)
        
    def forward(self, x_in, initial_hidden = None):
        '''
        Args
        x_in (tensor): batch_size * sequence_length * Embedding_size
        '''
        #print('input shape = ', x_in.shape)
            
        if self.batch_first:
            x_in = x_in.permute(1,0,2).to(x_in.device)
            #Now x_in (tensor): sequence_length * batch_size * Embedding_size
            #print('after modifications for batch_first ', x_in.shape)
        
        seq_length, batch_size, embedding_length = x_in.shape    
        
        #Chech if initial hidden state is provided. Else initialize it.
        if initial_hidden is None:
            initial_hidden = self.get_initial_hidden_state(batch_size)
         
        hidden_t = initial_hidden
        
        hidden_vectors = []
        
        #calculate hidden state vectors by passing through the seqence
        for word_batch in x_in:
            #print('sequence shape = ', word_batch.shape)
            hidden_t = self.rnncell(word_batch,hidden_t)
            hidden_vectors.append(hidden_t)
        
        #convert to pytorch hidden vectors
        hidden_vectors = torch.stack(hidden_vectors)
        
        #print('hidden vectors = ', hidden_vectors.shape)
        
        if self.batch_first:
            hidden_vectors = hidden_vectors.permute(1,0,2)
            #print('hidden vectors for batch_first ', hidden_vectors.shape)
            
        return hidden_vectors
        
```

### Create vectorizer class which generates vectors from surnames


```python
class Vectorizer():
    def __init__(self, surname_vocabulary, nationality_vocabulary):
        self.surname_vocabulary = surname_vocabulary
        self.nationality_vocabulary = nationality_vocabulary
    
    def vectorize(self, surname, vector_length = -1):
        indices = [self.surname_vocabulary.start_index]
        
        #TODO: handle unknown token
        indices.extend( [self.surname_vocabulary.token_to_idx[char] for char in surname] )
        indices.append(self.surname_vocabulary.last_index)
        
        if vector_length < 0:
            vector_length = len(indices)
        vector = np.zeros((vector_length,), dtype = np.int64)
        
        copy_length = min(vector_length,len(indices))
        
        vector[:copy_length] = indices[:copy_length]
        vector[copy_length:] = self.surname_vocabulary.mask_index
        
        return vector
    
    @classmethod
    def from_df(cls, surname_df):
        surnames = surname_df['surname'].values
        nationalities = surname_df['nationality'].values
        
        surname_vocabulary = SequenceVocabulary()
        nationality_vocabulary = Vocabulary()
        
        for surname in surnames:
            for char in surname:
                surname_vocabulary.add_token(char)
        
        for nat in nationalities:
            nationality_vocabulary.add_token(nat)
        
        return cls(surname_vocabulary, nationality_vocabulary)
            
```

### Vocabulary class to store index and tokens


```python
class Vocabulary():
    def __init__(self , add_unk = False):
        self.idx_to_token = {}
        self.token_to_idx = {}
        if add_unk:
            self.unknown_token = '##unk'
            self.add_token(self.unknown_token)
        
    def add_token(self,token):
        if token not in self.token_to_idx.keys():
            index = len(self.idx_to_token)
            self.idx_to_token[index] = token
            self.token_to_idx[token] = index
        return self.token_to_idx[token]
            
class SequenceVocabulary(Vocabulary):
    def __init__(self):
        super(SequenceVocabulary,self).__init__(True)

        self.mask_index = self.add_token('##mask')
        self.start_index = self.add_token('##first')
        self.last_index = self.add_token('##last')
```

### Pytorch dataset


```python
class SurnameDataset(Dataset):
    def __init__(self, vectorizer, dataframe):
        super(SurnameDataset,self).__init__()
        self.vectorizer = vectorizer
        self.df = dataframe
        self.df_active = self.df[self.df['split']=='train']
        
    def set_split(self, split):
        self.df_active = self.df[self.df['split']==split]
        
    def __getitem__(self, index):
        series = self.df_active.iloc[index]
        surname = series['surname']
        nationality = series['nationality']
        
        surname_indexed_vector = self.vectorizer.vectorize(surname,args.max_surname_len)
        nationality_index = self.vectorizer.nationality_vocabulary.token_to_idx[nationality]
        return {'x' : surname_indexed_vector,
               'y' : nationality_index}
    def __len__(self):
        return len(self.df_active)
    
    @classmethod
    def from_df(cls, df):
        vectorizer = Vectorizer.from_df(df)
        return cls(vectorizer,df)
```

### configs


```python
args = Namespace(
    surname_csv = r'data\surname\surnames_split_krishan.csv', #change this path
    epochs = 20,
    lr = 0.03,
    loss = nn.CrossEntropyLoss,
    max_surname_len = 15,
    hidden_size = 64,
    embedding_size = 16
)

#Find 95th percentile length 
lengths = df[df['split'] == 'train']['surname'].apply(lambda x: len(x))
print('lengths of surnames =\n',lengths.describe())
import math
_95thperc = lengths.sort_values().iloc[math.floor(len(lengths)*0.95)]
print('95th percentile length = ',_95thperc)
args.max_surname_len = _95thperc
```

    lengths of surnames =
     count    7680.000000
    mean        6.663021
    std         1.983061
    min         1.000000
    25%         5.000000
    50%         6.000000
    75%         8.000000
    max        17.000000
    Name: surname, dtype: float64
    95th percentile length =  10
    

### Load , analyze data, create data loder


```python
df = pd.read_csv(args.surname_csv)
dataset = SurnameDataset.from_df(df)
train_dataloader = DataLoader(dataset,32,True)


num_tokens = len(dataset.vectorizer.surname_vocabulary.idx_to_token) #embedding row size
num_classes = len(dataset.vectorizer.nationality_vocabulary.idx_to_token)
print ('tokens = {}, classes = {}'.format(num_tokens,num_classes ))
df.sample(5)
```

    tokens = 88, classes = 18
    




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
      <th>nationality</th>
      <th>split</th>
      <th>surname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9327</th>
      <td>Irish</td>
      <td>val</td>
      <td>Cleirigh</td>
    </tr>
    <tr>
      <th>8053</th>
      <td>Chinese</td>
      <td>train</td>
      <td>Chew</td>
    </tr>
    <tr>
      <th>10331</th>
      <td>Spanish</td>
      <td>test</td>
      <td>Félix</td>
    </tr>
    <tr>
      <th>5045</th>
      <td>Russian</td>
      <td>train</td>
      <td>Pitaevsky</td>
    </tr>
    <tr>
      <th>7498</th>
      <td>Japanese</td>
      <td>train</td>
      <td>Hyobanshi</td>
    </tr>
  </tbody>
</table>
</div>



###  Create Model



```python
class SurnameRNNClassifier(nn.Module):
    def __init__(self, feature_size, output_classes):
        super(SurnameRNNClassifier,self).__init__()
        
        self.embedding = nn.Embedding(feature_size, 
                                      args.embedding_size,
                                      dataset.vectorizer.surname_vocabulary.mask_index)
        self.rnn = ElmanRNN(args.embedding_size, args.hidden_size, True)
        
        self.linear = nn.Linear(args.hidden_size, output_classes)
        
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x_in, apply_softmax = False):
        '''
        args
        x_in = batch_size * sequence_length 
        '''
        #print('x shape = ',x_in.shape)
        yhat = self.embedding(x_in) # batch_size * sequence_length * embedding_dim
        
        #Functional DROPOUT
        #no idea whether it's train or eval. Use nn.Dropout
        #yhat = torch.nn.functional.dropout(yhat, 0.1)
        
        yhat = self.rnn(yhat) # batch_size * sequence_length(num hiddens) * hidden_dim
        
        yhat = yhat[:, -1, :]
        yhat = yhat.squeeze()
        #DROPOUT
        yhat = self.dropout(yhat)
        
        yhat = self.linear(yhat) # batch_size * output_classes

        if apply_softmax:
            yhat= torch.nn.Softmax(1)(yhat)# batch_size * output_classes

        return yhat
```

### Function to calculate validation loss and accuracy over entire split


```python
import pdb
def calculate_loss_acc(dataset,model,split='val'):
    #pdb.set_trace()
    dataset.set_split(split)
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    total = len(dataset)
    correct = 0
    losses = []
    
    model.eval()
    
    for data in dataloader:
        x = data['x']
        y = data['y']
        
        yhat = model(x)

        loss = args.loss()(yhat,y).item()
        losses.append(loss)
        
        yhat = torch.argmax(yhat,dim=1)
        correct += torch.sum(y==yhat).item()
    loss = sum(losses)/len(losses)
    acc = correct/total
    
    dataset.set_split('train')
    return loss, acc

```

###  Train



```python
model = SurnameRNNClassifier(num_tokens, num_classes )
print(model)
optimizer = torch.optim.SGD(model.parameters(),lr = args.lr)
train_losses, val_losses, val_accs = [],[],[]

for epoch in range(args.epochs):
    losses = []
    for data in train_dataloader:
        x = data['x']
        y = data['y']
        
        model.train()
        yhat = model(x)
        
        optimizer.zero_grad()
        
        loss = args.loss()(yhat,y)
        loss.backward()
        losses.append(loss.item())
        
        optimizer.step()
    train_losses.append(sum(losses)/len(losses))
    val_loss, val_acc = calculate_loss_acc(dataset, model)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print('epoch {} : train_loss {}, val_loss {}, val_acc {}'.format(epoch
          ,train_losses[-1],val_losses[-1],val_accs[-1]))
```

    SurnameRNNClassifier(
      (embedding): Embedding(88, 16, padding_idx=1)
      (rnn): ElmanRNN(
        (rnncell): RNNCell(16, 64)
      )
      (linear): Linear(in_features=64, out_features=18, bias=True)
      (dropout): Dropout(p=0.1)
    )
    epoch 0 : train_loss 2.2965063979228337, val_loss 2.142411273259383, val_acc 0.3591463414634146
    epoch 1 : train_loss 2.050973414381345, val_loss 1.9430594260875995, val_acc 0.4451219512195122
    epoch 2 : train_loss 1.907807615896066, val_loss 1.8554594562603877, val_acc 0.45670731707317075
    epoch 3 : train_loss 1.8028959915041924, val_loss 1.7653987591083233, val_acc 0.4865853658536585
    epoch 4 : train_loss 1.7090267524123193, val_loss 1.6766792031434865, val_acc 0.5140243902439025
    epoch 5 : train_loss 1.6379290473957857, val_loss 1.6299713666622455, val_acc 0.5219512195121951
    epoch 6 : train_loss 1.5674623471995195, val_loss 1.5803549518952003, val_acc 0.5323170731707317
    epoch 7 : train_loss 1.5027624413371086, val_loss 1.508364567389855, val_acc 0.5670731707317073
    epoch 8 : train_loss 1.4425324335694314, val_loss 1.4327852405034578, val_acc 0.5804878048780487
    epoch 9 : train_loss 1.382294626533985, val_loss 1.4343391840274518, val_acc 0.5676829268292682
    epoch 10 : train_loss 1.325134468326966, val_loss 1.3726348556005037, val_acc 0.6060975609756097
    epoch 11 : train_loss 1.2833637627462546, val_loss 1.3074743701861455, val_acc 0.6067073170731707
    epoch 12 : train_loss 1.2457967475056648, val_loss 1.2868399597131288, val_acc 0.6225609756097561
    epoch 13 : train_loss 1.2039708423117796, val_loss 1.2699937522411346, val_acc 0.6341463414634146
    epoch 14 : train_loss 1.1685669176280498, val_loss 1.2110394858396971, val_acc 0.650609756097561
    epoch 15 : train_loss 1.1382594662408034, val_loss 1.2288372333233173, val_acc 0.6603658536585366
    epoch 16 : train_loss 1.1155679715176423, val_loss 1.1853035642550542, val_acc 0.6670731707317074
    epoch 17 : train_loss 1.0854565724730492, val_loss 1.141878836430036, val_acc 0.676219512195122
    epoch 18 : train_loss 1.0628218047320843, val_loss 1.181883229659154, val_acc 0.6554878048780488
    epoch 19 : train_loss 1.037904354929924, val_loss 1.1669966051211724, val_acc 0.6731707317073171
    


```python
import matplotlib.pyplot as pp
pp.xlabel('epochs')
pp.title('losses')
pp.plot(train_losses)
pp.plot(val_losses)
pp.legend(['train','val'])
pp.figure()
pp.plot(val_accs)
pp.title('validation accuracy')
pp.show()
```


![loss](/assets/rnn/loss.png)
![acc](/assets/rnn/acc.png)

```python
test_loss,test_acc = calculate_loss_acc(dataset,model,'test')
test_loss,test_acc
```

    (1.2183352983914888, 0.6608433734939759)


