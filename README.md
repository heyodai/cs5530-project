# CS 5530 - Team "Front of the Class"

This is the repository for our team's CS 5530 project. The goal of this project is to create a text summarizer using deep learning. The model will be trained on a dataset of news articles and will be able to generate summaries for new articles.

## Table of Contents

- [Team Members](#team-members)
- [Dataset](#dataset)
- [Training Process/Roadmap](#training-processroadmap)
- [Results](#results)

## Team Members

- [Odai Athamneh](https://github.com/heyodai)
- [Devin Cline](https://github.com/orangedoor)
- [Michael Nweke](https://github.com/m-nweke)
- [Feng Zheng](https://github.com/FengZheng99)

## Dataset

The training dataset is a collection of news articles from 3 separate datasets (XSum, CNN/Daily Mail, Multi-News). The dataset contains 870,521 articles, each of which has a text content and a summary. The text content is the full article, and the summary is a short version of the article that summarizes the main points. 

The dataset is available [on Kaggle](https://www.kaggle.com/datasets/sbhatti/news-summarization) and is licensed under a [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) license. The dataset is also available on the [Hugging Face Datasets](https://huggingface.co/datasets/cnn_dailymail) library.

We do not include the dataset in this repository because it is too large. However, you can download the dataset from the links above and place it in the `data` directory.

The dataset contains the following columns:
- `Unnamed: 0` - index, can be ignored
- `ID` - unique ID for each article (appears to be a generated UUID)
- `Content` - the text content of the article
- `Summary` - the summary of the article
- `Dataset` - the dataset that the article came from (XSum, CNN/Daily Mail, Multi-News)

## Training Process/Roadmap

This is a rough outline of the steps that we will take to train the model:

- **Phase 1 - Data Acquisition and Refinement** (see `phase_1.ipynb`)
    - [x] Obtain a dataset of news articles that includes the text content as well as a summary of each article. 
    - [x] Explore the dataset to get a sense of the data, such as the number of articles, length of the articles and summaries, and distribution of topics and keywords.
    - [x] Clean and preprocess the data to remove unnecessary characters, punctuation, and stop words. 
    - [x] Tokenize the text into words or subwords, and create input sequences and output summaries.
- **Phase 2 - Model Architecture Selection** (see `phase_2.ipynb`)
    - [x] Choose a deep learning model that is suitable for the task of summarization. 
        - We will most likely go with a transformer-based model, such as BERT, T5, GPT-J, or GPT-Neo.
        - However, we could also look into an encoder-decoder model or a pointer-generator network. 
    - [x] Define the architecture of the model, including the number of layers, hidden units, and embedding dimensions.
- **Phase 3 - Model Training and Evaluation** 
    - [x] Split the dataset into training, validation, and test sets.
    - [x] Train the model on the training set and monitor its performance on the validation set. 
        - [x] Use techniques like early stopping and learning rate annealing to prevent overfitting and improve performance. 
    - [x] Evaluate the model on the test set to get a final estimate of its performance.
- **Phase 4 - Iterative Improvements** 
    - [x] Finally, analyze the performance of the model and identify areas for improvement. 
    - [x] This could involve collecting more data, tuning hyperparameters, or experimenting with different architectures. 
    - [x] Iterate through these steps until the model achieves satisfactory performance.

## Results

The table below contains our inital accuracy scores. Note that: 

- For WMD scores, a lower number indicates that the two inputs are more similar. WMD measures the distance between the two sets of word embeddings, so a lower distance indicates that the two sets are more similar, or require fewer steps to be transformed into one another.
- For SIF scores, a higher number indicates that the two inputs are more similar. SIF measures the cosine similarity between the two sets of word embeddings, so a higher cosine similarity indicates that the two sets are more similar.

Therefore, a lower WMD score and a higher SIF score both indicate that the two inputs are more similar.

|            | WMD Mean   | SIF Mean |
|------------|------------|----------|
| DistilBERT | 4.4856     | 0.8400   |
| MobileBERT | 2204752.00 | 0.9992   |
| ALBERT     | 15.5492    | 0.8180   |
| ELECTRA    | 8.3385     | 0.3745   |

MobileBERT performed significantly better by WMD score than the other models, while ELECTRA performed the best by SIF score. This suggests that MobileBERT is better at capturing the semantic similarities between the text inputs, while ELECTRA is better at capturing the overall meaning of the text.