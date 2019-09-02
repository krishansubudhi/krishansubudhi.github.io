---
author: krishan
layout: post
categories: deeplearning
title: Introduction to Transformers
---

# What are Transformers?

Transformers are new state-of-the-art deep learning architectures used for Natural Language Processing(NLP).

# What is NLP?

NLP comprises of AI tasks related to text processing. Some example are - machine translation, text classification (spam, focused inbox), text generation (smart reply, smart compose) etc.

# What are other deep learning architectures used for NLP?

The traditional architectures consist of Dense Networks, Recurrent Neural Network, 1 D CNN, RNN with attention etc. LSMT and GRU are variants of RNN.

# How is Transformer different than the rest?

RNN reads text in one direction; sort of like reading a line from left to right or right to left. The problem is, for longer sequences the context is usually lost by the time the model reads the last word. LSTM and GRU solve this problem to some extent. But for complex tasks involving long range dependencies , the RNN architecture is not capable of producing good results.

1 D CNN reads text one chunk at a time, processes them separately and combines the results. This technique is adopted from image processing. The drawback here is - the intricate dependencies between the chunks are lost.

Transformers, however, read all the words at once. It uses a mechanism called self attention which allows the model to go back to a particular word when necessary. This trait of going back is learned by training the transformer models on huge amount of data. Since it does not use temporal features like RNN, transformer training capitalizes on parallel processing. This makes the training faster and builds a better model in less time.

# What's the catch?

The problems with transformers is it's fixed input length, high memory footprint and latency. Model distillation is a technique used to overcome the memory and latency issues.

# What's next?

Research is going on to accommodate even longer dependencies in transformers. Transformer XL , which has a caching mechanism to reuse previous results, is the latest buzz in NLP world.

