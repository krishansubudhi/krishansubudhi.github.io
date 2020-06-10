---
comments: true
author: krishan
layout: post
categories: deeplearning
title: Type faster using RoBERTA 
description: Typing fast on phones are prone to error. Using masked language models, these errors can be detected and rectified
---

The goal of the experiment is to detect and correct the mistakes during fast typing on phone while using the swipe feature. Fast gestures in swipe currently produce some wrong results and there is no flagging/correction done after a sentence is typed. User has to go back and check correctness or reduce the swiping speed. Using language models we can detect the mistakes and improve the typing speed.

### Setup
I typed few sentences using google keyboad ans swiftkey as fast as i could and found 80-90% of the words were correct. But the rest did not make sense in the context of the sentence. Using [Roberta Masked Language model](https://huggingface.co/transformers/model_doc/roberta.html), those errors can be detected and rectified after a sentence has been typed. This feature can be used in phone keyboards as a second layer of check after a sentence is typed.

### Algorithm
Mask each word in the raw sentence and pass it to roberta model. Collect the labels find the probability of the masked word , top 3 suggestions and their probabilities. If a sentence has N tokens, then N forward passes are done.


```python

from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
```


```python
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaForMaskedLM.from_pretrained('roberta-base')
```


```python

def autocorrect(sentence):
    tokens = tokenizer.tokenize(sentence,add_prefix_space=False)
    #print(tokens)
    inputs = tokenizer.encode(sentence, add_special_tokens=True, add_prefix_space=False)
    labels = torch.tensor([1]).unsqueeze(0)

    #mask each token and find prediction
    final_scores = []
    from tqdm.notebook import tqdm
    for i,tok_id in tqdm(enumerate(inputs),):
        input_masked = inputs.copy()
        input_masked[i] = tokenizer.mask_token_id
        input_ids = torch.tensor(input_masked).unsqueeze(0)
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]
        final_scores.append(prediction_scores.squeeze()[i])
    prediction_scores = torch.stack(final_scores)
    prediction_scores = prediction_scores.softmax(dim = -1)

    #convert a list of ids to list of words. Removes special character from BPE tokenizer.
    convert_to_words = lambda inputs : \
    [tokenizer.convert_tokens_to_string(tok).strip() for tok in tokenizer.convert_ids_to_tokens(inputs)]

    #convert_to_words = lambda inputs : list(inputs)

    probs_list = prediction_scores.sort(descending = True)
    indexes_list = probs_list.indices.squeeze()[:,:3]
    values_list = probs_list.values.squeeze()[:,:3]
    # print(probs_list)
    original_prob = ["%.4f" %prediction_scores.squeeze()[pos,index] for pos, index in enumerate(inputs)]
    options = [convert_to_words(indexes) for indexes in indexes_list]
    values_list = [["%.3f" % v for v in vals] for vals in values_list.tolist()]
    original = convert_to_words(inputs)

    #display results using pandas dataframe
    import pandas as pd
    print(f'Sentence = {tokenizer.decode(inputs)}')
    return pd.DataFrame({'original_text':original, 'original_prob':original_prob, 'suggestions':options, 'suggestions_prob':values_list,}).head(100)
```


```python
sentences = ["I am trying to write a text using drawer which is sometimes wing." ,
"Two words were wrongly typed here. More I I will need to go back to reach quoted and correct then. ",
"There will be probability calculated for each word which will decide whether the word is appropriate at that place. If not, it will either be replaced or deleted. Special checks to handle succeed tons." ,
"I implemented the salary auto correct algorithm I wanted. It is identifying mistakes but the suggestions are not what I wanted. It probably needs fine-tuning on my father. "]

df = autocorrect(sentences[0])
df
```

  Sentence = \<s>I am trying to write a text using drawer which is sometimes wing.\</s>





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
      <th>original_text</th>
      <th>original_prob</th>
      <th>suggestions</th>
      <th>suggestions_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;s&gt;</td>
      <td>1.0000</td>
      <td>[&lt;s&gt;, ., &lt;/s&gt;]</td>
      <td>[1.000, 0.000, 0.000]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I</td>
      <td>0.9914</td>
      <td>[I, i, I]</td>
      <td>[0.991, 0.007, 0.000]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>am</td>
      <td>0.2943</td>
      <td>['m, am, was]</td>
      <td>[0.463, 0.294, 0.214]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>trying</td>
      <td>0.6924</td>
      <td>[trying, able, going]</td>
      <td>[0.692, 0.094, 0.047]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>to</td>
      <td>0.9995</td>
      <td>[to, and, the]</td>
      <td>[0.999, 0.000, 0.000]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>write</td>
      <td>0.0959</td>
      <td>[send, type, write]</td>
      <td>[0.323, 0.107, 0.096]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>a</td>
      <td>0.1075</td>
      <td>[the, this, some]</td>
      <td>[0.361, 0.140, 0.134]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>text</td>
      <td>0.0042</td>
      <td>[story, novel, poem]</td>
      <td>[0.230, 0.152, 0.094]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>using</td>
      <td>0.0009</td>
      <td>[for, book, about]</td>
      <td>[0.129, 0.070, 0.067]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>drawer</td>
      <td>0.0000</td>
      <td>[html, HTML, JavaScript]</td>
      <td>[0.119, 0.096, 0.074]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>which</td>
      <td>0.2974</td>
      <td>[that, which, it]</td>
      <td>[0.443, 0.297, 0.054]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>is</td>
      <td>0.1893</td>
      <td>[I, is, can]</td>
      <td>[0.451, 0.189, 0.089]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>sometimes</td>
      <td>0.0004</td>
      <td>[a, my, very]</td>
      <td>[0.145, 0.136, 0.061]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>wing</td>
      <td>0.0000</td>
      <td>[difficult, hard, tricky]</td>
      <td>[0.295, 0.150, 0.045]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>.</td>
      <td>0.7074</td>
      <td>[., :, !]</td>
      <td>[0.707, 0.078, 0.028]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>&lt;/s&gt;</td>
      <td>1.0000</td>
      <td>[&lt;/s&gt;, I, (]</td>
      <td>[1.000, 0.000, 0.000]</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = autocorrect(sentences[1])
df
```

Sentence = \<s>Two words were wrongly typed here. More I I will need to go back to reach quoted and correct then.\</s>





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
      <th>original_text</th>
      <th>original_prob</th>
      <th>suggestions</th>
      <th>suggestions_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;s&gt;</td>
      <td>1.0000</td>
      <td>[&lt;s&gt;, ., &lt;/s&gt;]</td>
      <td>[1.000, 0.000, 0.000]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Two</td>
      <td>0.0267</td>
      <td>[Some, Many, Several]</td>
      <td>[0.559, 0.113, 0.057]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>words</td>
      <td>0.2290</td>
      <td>[words, sentences, lines]</td>
      <td>[0.229, 0.185, 0.155]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>were</td>
      <td>0.2506</td>
      <td>[are, were, I]</td>
      <td>[0.465, 0.251, 0.212]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>wrongly</td>
      <td>0.0491</td>
      <td>[not, incorrectly, already]</td>
      <td>[0.245, 0.133, 0.090]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>typed</td>
      <td>0.0025</td>
      <td>[quoted, used, written]</td>
      <td>[0.597, 0.067, 0.045]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>here</td>
      <td>0.1820</td>
      <td>[here, in, there]</td>
      <td>[0.182, 0.120, 0.109]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>.</td>
      <td>0.8207</td>
      <td>[., ,, and]</td>
      <td>[0.821, 0.050, 0.045]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>More</td>
      <td>0.0001</td>
      <td>[So, Which, I]</td>
      <td>[0.447, 0.044, 0.044]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I</td>
      <td>0.0000</td>
      <td>[., so, ,]</td>
      <td>[0.398, 0.061, 0.049]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>I</td>
      <td>0.0028</td>
      <td>[think, guess, suspect]</td>
      <td>[0.455, 0.111, 0.048]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>will</td>
      <td>0.4178</td>
      <td>[will, 'll, may]</td>
      <td>[0.418, 0.256, 0.054]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>need</td>
      <td>0.1014</td>
      <td>[have, need, try]</td>
      <td>[0.848, 0.101, 0.030]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>to</td>
      <td>0.9965</td>
      <td>[to, and, will]</td>
      <td>[0.997, 0.001, 0.000]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>go</td>
      <td>0.7911</td>
      <td>[go, get, come]</td>
      <td>[0.791, 0.020, 0.019]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>back</td>
      <td>0.7227</td>
      <td>[back, through, further]</td>
      <td>[0.723, 0.070, 0.019]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>to</td>
      <td>0.1531</td>
      <td>[and, ,, to]</td>
      <td>[0.589, 0.161, 0.153]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>reach</td>
      <td>0.0000</td>
      <td>[the, be, them]</td>
      <td>[0.229, 0.090, 0.072]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>quoted</td>
      <td>0.0000</td>
      <td>[them, out, back]</td>
      <td>[0.114, 0.083, 0.035]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>and</td>
      <td>0.0653</td>
      <td>[it, word, them]</td>
      <td>[0.167, 0.107, 0.104]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>correct</td>
      <td>0.0497</td>
      <td>[write, correct, post]</td>
      <td>[0.057, 0.050, 0.042]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>then</td>
      <td>0.0001</td>
      <td>[them, it, this]</td>
      <td>[0.332, 0.321, 0.054]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>.</td>
      <td>0.8164</td>
      <td>[., :, !]</td>
      <td>[0.816, 0.095, 0.019]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>&lt;/s&gt;</td>
      <td>0.9997</td>
      <td>[&lt;/s&gt;, ., "]</td>
      <td>[1.000, 0.000, 0.000]</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusion
Evident from the result, The Roberta model is very confident on the mistake positions during typing. Tokens with very low probability (0.0000) are definitely incorrect.

The suggestions are also good but not great. Finetuning on my keyboard inputs will make the suggestions better. Also masking whole words instead of just tokens might help removing any bias because pieces of wrong words might influence the results.



```python

```
