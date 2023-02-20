# CS 5530 - Team "Front of the Class"

This is the repository for our team's CS 5530 project. The goal of this project is to create a text summarizer using deep learning. The model will be trained on a dataset of news articles and will be able to generate summaries for new articles.

## Table of Contents

- [Team Members](#team-members)
- [Dataset](#dataset)
- [Training Process/Roadmap](#training-processroadmap)
- [Results](#results)
- [Setup](#setup)

## Team Members

- [Odai Athamneh](https://github.com/heyodai)
- [Devin Cline](https://github.com/orangedoor)
- [Michael Nweke](https://github.com/m-nweke)
- [Feng Zheng](https://github.com/FengZheng99)

## Dataset

The training dataset is a collection of news articles from 3 separate datasets (XSum, CNN/Daily Mail, Multi-News). The dataset contains 870,521 articles, each of which has a text content and a summary. The text content is the full article, and the summary is a short version of the article that summarizes the main points. 

The dataset is available [on Kaggle](https://www.kaggle.com/datasets/sbhatti/news-summarization) and is licensed under a [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) license. The dataset is also available on the [Hugging Face Datasets](https://huggingface.co/datasets/cnn_dailymail) library.

## Training Process/Roadmap

This is a rough outline of the steps that we will take to train the model:

1. **Data acquisition and exploration:** Obtain a dataset of news articles that includes the text content as well as a summary of each article. Explore the dataset to get a sense of the data, such as the number of articles, length of the articles and summaries, and distribution of topics and keywords.
2. **Data preprocessing:** Clean and preprocess the data to remove unnecessary characters, punctuation, and stop words. Tokenize the text into words or subwords, and create input sequences and output summaries.
3. **Data splitting:** Split the dataset into training, validation, and test sets. The training set is used to train the model, the validation set is used to tune hyperparameters and monitor the model's performance, and the test set is used to evaluate the model's performance on unseen data.
4. **Model selection and architecture:** Choose a deep learning model that is suitable for the task of summarization. This could be an encoder-decoder model, a transformer-based model, or a pointer-generator network. Define the architecture of the model, including the number of layers, hidden units, and embedding dimensions.
5. **Model training and evaluation:** Train the model on the training set and monitor its performance on the validation set. Use techniques like early stopping and learning rate annealing to prevent overfitting and improve performance. Evaluate the model on the test set to get a final estimate of its performance.
7. **Iterative improvement:** Finally, analyze the performance of the model and identify areas for improvement. This could involve collecting more data, tuning hyperparameters, or experimenting with different architectures. Iterate through these steps until the model achieves satisfactory performance.

## Results

## Setup