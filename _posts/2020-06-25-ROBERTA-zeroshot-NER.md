---
comments: true
author: krishan
layout: post
categories: deeplearning
title: Zero shot NER using RoBERTA
description: Using Roberta last layer embedding and cosine similarity, NER can be performed in a zero shot manner. The model performance is very good without any training. This notebooks finds similar entities given an example entity.
---

```python
import torch
import transformers
```


```python
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-base')
```


```python


def tokenize(txt):
    segments = txt.split('@')
    tokenized_segments = [tokenizer.tokenize(segment) for segment in segments]
    positions = [len(s) for s in tokenized_segments]
    positions = [positions[i]+positions[i-1] if i>0 else positions[i] for i in range(len(positions)) ]
    
    final_tokens = []
    for tokens in tokenized_segments:
        final_tokens.extend(tokens)
    return final_tokens, positions

def merge_embeddings(token_embeddings, start, end, operation = torch.mean):
    #print('token_embedding_to merge',token_embeddings[start:end].shape, token_embeddings[start:end][:,:5])
    merged =  operation(token_embeddings[start:end], dim = 0)
    #print('after merging',merged.shape, merged[:5])
    return merged

# merged = merge_embeddings(token_embeddings[start:end], input[1][0],input[1][1])
# print(merged.shape, merged[:5])
# return merged

def get_embeddings(txt, example = False):
    input = tokenize(txt)
    print(input[0])
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(input[0])).unsqueeze(0)
    op = model(input_ids)
    if example:
        return input[0] , op[0].data.squeeze(), input[1][0], input[1][1]
    else:
        return input[0], op[0].data.squeeze()
    
```


```python
example = 'Eighteen relatives test positive for @coronavirus@ after surprise birthday party'
test_text = 'Diagnosing Ebola virus disease (EVD) shortly after infection can be difficult. Early symptoms of EVD such as fever, headache, and weakness are not specific to Ebola virus infection and often are seen in patients with other more common diseases, like malaria and typhoid fever.'

def find_entities(example, test_text):
    example_tokens, embeddings, start, end = get_embeddings(example, True)
    entity_embedding = merge_embeddings(embeddings, start, end)



    test_tokens , test_embeddings= get_embeddings(test_text)
    similarity = F.cosine_similarity(test_embeddings, entity_embedding , dim = -1)
    print(similarity)
    max_similarity_index = torch.argmax(similarity)
    print()
    print('Example = ',example)
    print('Test sentence = ', test_text)
    print('Most similar entity  = to {} is {}'.format( example_tokens[start:end], test_tokens[max_similarity_index]))



```


```python
example = 'Eighteen relatives test positive for @coronavirus@ after surprise birthday party'
test_text = 'Diagnosing Ebola virus disease (EVD) shortly after infection can be difficult. Early symptoms of EVD such as fever, headache, and weakness are not specific to Ebola virus infection and often are seen in patients with other more common diseases, like malaria and typhoid fever.'

find_entities(example, test_text)
```

    ['Eight', 'een', 'Ġrelatives', 'Ġtest', 'Ġpositive', 'Ġfor', 'cor', 'on', 'av', 'irus', 'Ġafter', 'Ġsurprise', 'Ġbirthday', 'Ġparty']
    ['Di', 'agn', 'osing', 'ĠEbola', 'Ġvirus', 'Ġdisease', 'Ġ(', 'E', 'VD', ')', 'Ġshortly', 'Ġafter', 'Ġinfection', 'Ġcan', 'Ġbe', 'Ġdifficult', '.', 'ĠEarly', 'Ġsymptoms', 'Ġof', 'ĠEV', 'D', 'Ġsuch', 'Ġas', 'Ġfever', ',', 'Ġheadache', ',', 'Ġand', 'Ġweakness', 'Ġare', 'Ġnot', 'Ġspecific', 'Ġto', 'ĠEbola', 'Ġvirus', 'Ġinfection', 'Ġand', 'Ġoften', 'Ġare', 'Ġseen', 'Ġin', 'Ġpatients', 'Ġwith', 'Ġother', 'Ġmore', 'Ġcommon', 'Ġdiseases', ',', 'Ġlike', 'Ġmalaria', 'Ġand', 'Ġtyph', 'oid', 'Ġfever', '.']
    tensor([0.6868, 0.7599, 0.7937, 0.8389, 0.8257, 0.8205, 0.7709, 0.7793, 0.7412,
            0.7899, 0.7685, 0.7957, 0.8104, 0.7854, 0.7828, 0.7782, 0.6981, 0.7919,
            0.7918, 0.8179, 0.7722, 0.7714, 0.7975, 0.7852, 0.7869, 0.7720, 0.7952,
            0.7762, 0.7822, 0.7808, 0.7863, 0.8046, 0.7516, 0.7595, 0.8209, 0.8285,
            0.7982, 0.7687, 0.7613, 0.7931, 0.7817, 0.7899, 0.7919, 0.7848, 0.7764,
            0.7930, 0.7979, 0.7966, 0.7894, 0.7748, 0.8245, 0.7462, 0.8360, 0.8105,
            0.8380, 0.6981])
    
    Example =  Eighteen relatives test positive for @coronavirus@ after surprise birthday party
    Test sentence =  Diagnosing Ebola virus disease (EVD) shortly after infection can be difficult. Early symptoms of EVD such as fever, headache, and weakness are not specific to Ebola virus infection and often are seen in patients with other more common diseases, like malaria and typhoid fever.
    Most similar entity  = to ['cor', 'on', 'av', 'irus'] is ĠEbola
    


```python
example = 'It is too @hot@ today'
test_text = 'The jug was very cold when I held it'

find_entities(example, test_text)
```

    ['It', 'Ġis', 'Ġtoo', 'hot', 'Ġtoday']
    ['The', 'Ġjug', 'Ġwas', 'Ġvery', 'Ġcold', 'Ġwhen', 'ĠI', 'Ġheld', 'Ġit']
    tensor([0.7786, 0.8400, 0.8469, 0.8501, 0.8804, 0.7686, 0.8470, 0.8215, 0.8569])
    
    Example =  It is too @hot@ today
    Test sentence =  The jug was very cold when I held it
    Most similar entity  = to ['hot'] is Ġcold
    


```python
example = 'I am going to the marked to bring @groceries@'
test_text = 'She found vegetables in her fridge'

find_entities(example, test_text)
```

    ['I', 'Ġam', 'Ġgoing', 'Ġto', 'Ġthe', 'Ġmarked', 'Ġto', 'Ġbring', 'gro', 'cer', 'ies']
    ['She', 'Ġfound', 'Ġvegetables', 'Ġin', 'Ġher', 'Ġfridge']
    tensor([0.7984, 0.8804, 0.8919, 0.8646, 0.8773, 0.8163])
    
    Example =  I am going to the marked to bring @groceries@
    Test sentence =  She found vegetables in her fridge
    Most similar entity  = to ['gro', 'cer', 'ies'] is Ġvegetables
    


```python
example = '@Microsoft@ is a trillion dollar company'
test_text = 'Apple surpassed in valuation last year'

find_entities(example, test_text)
```

    ['Microsoft', 'Ġis', 'Ġa', 'Ġtrillion', 'Ġdollar', 'Ġcompany']
    ['Apple', 'Ġsurpassed', 'Ġin', 'Ġvaluation', 'Ġlast', 'Ġyear']
    tensor([0.9918, 0.7198, 0.7624, 0.7723, 0.6974, 0.7252])
    
    Example =  @Microsoft@ is a trillion dollar company
    Test sentence =  Apple surpassed in valuation last year
    Most similar entity  = to ['Microsoft'] is Apple
    


```python
example = '@India@ is seventh largest state in the world'
test_text = '@Australia@ is also a continent'

find_entities(example, test_text)
```

    ['India', 'Ġis', 'Ġseventh', 'Ġlargest', 'Ġstate', 'Ġin', 'Ġthe', 'Ġworld']
    ['Australia', 'Ġis', 'Ġalso', 'Ġa', 'Ġcontinent']
    tensor([0.9963, 0.8676, 0.7992, 0.8006, 0.8450])
    
    Example =  @India@ is seventh largest state in the world
    Test sentence =  @Australia@ is also a continent
    Most similar entity  = to ['India'] is Australia
    


```python
example = 'I like @Coffee@ in the morning'
test_text = 'We used to get free tea in our office'

find_entities(example, test_text)
```

    ['I', 'Ġlike', 'C', 'off', 'ee', 'Ġin', 'Ġthe', 'Ġmorning']
    ['We', 'Ġused', 'Ġto', 'Ġget', 'Ġfree', 'Ġtea', 'Ġin', 'Ġour', 'Ġoffice']
    tensor([0.7623, 0.8214, 0.8334, 0.8494, 0.8516, 0.9055, 0.8505, 0.8591, 0.8683])
    
    Example =  I like @Coffee@ in the morning
    Test sentence =  We used to get free tea in our office
    Most similar entity  = to ['C', 'off', 'ee'] is Ġtea
    


```python

```
