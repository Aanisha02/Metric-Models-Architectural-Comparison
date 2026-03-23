# Metric Models for Detection of LLM Texts

This repository contains our implementation of a metric learning framework for the task of identifying whether texts are written by human or AI. 

- The initial results of this work were presented in ICLR 2024 - 5th Workshop on practical ML for limited/low resource settings
- The full version of this paper will be published on ACM Transactions on Management of Information System very soon

The codes were originally developed on Google Colab with each experiment being completely implemented in a notebook. There are currently a lot of code duplications. We will modularize the whole repository in the near future.

## Introduction

In this paper, we propose a metric learning framework for LLM content detection that is balanced among computational costs, accessibility, and performance. 
Due to the generation technologies, LLMs tend to produce similar phrases in texts when receiving the same contexts. 
Exploiting this, the detection methods will rely on learning to compare a given text to an AI-generated reference from LLMs.
More specifically, the framework is trained to signify the similarity between LLM responses while decreasing that between LLM and human responses with triplet learning or contrastive learning combined with a same-context sampling strategy.
During the decision-making phase, the context of a given text is first prompted to a LLM to obtain a LLM-reference. 
The text and the LLM-equivalence are then fed to a metric framework to obtain their similarity metric. Finally, the metric is compared against a selected threshold to determine the text origination.

![image](https://github.com/user-attachments/assets/8fd3249b-f663-4e5c-aba2-cf7b5d6f48ec)

## Architecture

The complete framework comprises
1. A text embedding model that takes raw texts then transform to vector representations
2. A metric model to compute distance/similarity from the vectors

We currently MPNet as the embedding model. As MPNet can operate at either the full-text granularity or the sentence granularity, we develop two designs for metric models:
1. At full-text granularity: stacking Feed Forward blocks
2. At sentence granularity: transformer-like architecture

![image](https://github.com/user-attachments/assets/b4481661-ae20-4865-afd1-586f20d9e2cf)

## Benchmark data

For benchmarking, we collected over 100,000 context-content units to form five corpora where each instance consists of a context and three responses, one from human and two from LLMs.

![image](https://github.com/user-attachments/assets/346002e5-6469-4e27-9f74-dfec826fe701)

## Results

Our frameworks maintain F1 scores in between 0.87 to 0.95 across the tested corpora in multiple experiment settings. 
The metric framework also demands significantly less time in training and inference compared to the RoBERTa, LLaMA 3, Mistral v0.3, and Ghostbuster, while keeping 90% to 150% performances of the best benchmark.



