---
author: krishan
layout: post
categories: deeplearning
title: Word analogy using Glove Embeddings
---
# Word Embeddings

Load word embedding file into memory. Find analogy between different words based on word embedding


```python
import numpy as np
import annoy as nn
import os

from argparse import Namespace
```


```python
args = Namespace(
    embedding_folder = r'C:\Users\krkusuk\Downloads\glove.6B',
    embedding_file = r'glove.6B.50d.txt'
)
```


```python
class PretrainedEmbeddings(object):
    def __init__(self, word_to_index, word_vectors):
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        
        self.index_to_word = {v:k for k,v in self.word_to_index.items()}
        
        #nearest neighbour index
        self.index = nn.AnnoyIndex(len(word_vectors[0]), metric = 'euclidean')
        for _,i in self.word_to_index.items():
            self.index.add_item(i, self.word_vectors[i])
        #50 in number of trees. More trees, more precission
        self.index.build(50)
        
    @classmethod
    def from_embedding_file(cls, filepath):
        word_to_index={}
        word_vectors = []
        
        with open(filepath,encoding='UTF8') as f:
            for line in f.readlines():
                cols= line.split(' ')
                word = cols[0]
                embedding = np.array([float(x) for x in cols[1:]])
                word_to_index[word] = len(word_to_index)
                word_vectors.append(embedding)
        return cls(word_to_index, word_vectors)
    
    def get_word_vector(self, word):
        return self.word_vectors[self.word_to_index[word]]
    def get_nearest_neighbout(self, embedding, n =1):
        nn_indices = self.index.get_nns_by_vector(embedding, n)
        return [self.index_to_word[i] for i in nn_indices]
```

### Load word embedding from file

Link to download glove embedding 
http://nlp.stanford.edu/data/glove.6B.zip


```python
file = os.path.join(args.embedding_folder, args.embedding_file)
embedding = PretrainedEmbeddings.from_embedding_file(file)
```

## Find word analogy


```python
def find_analogy(w1,w2,w3, n):
    #w1-w2 = w3-w4
    #w4 = w3 + w2 -w1
    e1 = embedding.get_word_vector(w1)
    e2 = embedding.get_word_vector(w2)
    e3 = embedding.get_word_vector(w3)
    e4 = e3+e2-e1
    
    nearest_words = embedding.get_nearest_neighbout(e4,n)
    nearest_words = [word for word in nearest_words if word != w3]
    for i in range(len(nearest_words)):
        print('{} > {}:{} :: {}:{}'.format(i+1, w1,w2,w3,nearest_words[i]))
```


```python
wordsets =[
    ['cat','dog','tiger'],
    ['practical','impractical','freedom'],
    ['love','fight','marriage'],
    ['usa','mexico','india'],
]

for wordset in wordsets:
    print()
    find_analogy(wordset[0],wordset[1],wordset[2], 5)
```

    
    1 > cat:dog :: tiger:hunt
    2 > cat:dog :: tiger:hunting
    3 > cat:dog :: tiger:woods
    4 > cat:dog :: tiger:horse
    
    1 > practical:impractical :: freedom:declaring
    2 > practical:impractical :: freedom:proclaimed
    3 > practical:impractical :: freedom:traitor
    4 > practical:impractical :: freedom:powerless
    5 > practical:impractical :: freedom:traitors
    
    1 > love:fight :: marriage:ruled
    2 > love:fight :: marriage:rule
    3 > love:fight :: marriage:immunity
    4 > love:fight :: marriage:abortion
    5 > love:fight :: marriage:civil
    
    1 > usa:mexico :: india:province
    2 > usa:mexico :: india:provinces
    3 > usa:mexico :: india:indonesia
    4 > usa:mexico :: india:thailand
    5 > usa:mexico :: india:cambodia
    
