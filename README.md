# CS 5530 - Team "Front of the Class"

This is the repository for our team's CS 5530 project. The goal of this project was to create a news article text summarizer using deep learning. We explored the creation of a text summarizer using a transformer-based model, and we experimented with several different models to see which one performed the best.

## Table of Contents

- [Submission Overview](#submission-overview)
- [Environment Setup](#environment-setup)
- [Team Members](#team-members)
- [Dataset](#dataset)
- [Training Process/Roadmap](#training-processroadmap)
- [Results](#results)

## Submission Overview

1. **Source code**
    - We split the project work into 4 phases, and each phase has its own notebook. 
    - The notebooks are located in the `notebooks` directory. 
    - The *"Training Process/Roadmap"* section below contains a more detailed description of each phase.
2. **Datasets**
    - We did not include the datasets on GitHub because they are too large.
    - To reproduce our results, create a directory called `data` in the root of the repository, then download the following:
        - [raw.csv](https://drive.google.com/file/d/1wPYWk5mfTO3MuymdG4ZmcCcQfdx4W-GG/view?usp=sharing) - This is the raw dataset from Kaggle. See the *"Dataset"* section below for more information.
        - [cleaned.csv](https://drive.google.com/file/d/1X_bgh4v4LHWivnlRTZtQlA8N2b9lgcjt/view?usp=share_link) - This is the cleaned dataset that we used for training. See the *"Phase 1 - Data Acquisition and Refinement"* section below for more information.
3. **PowerPoint deck** - The PowerPoint presentation that we used in class is the `Presentation.pptx` file in the root of the repository.
4. **README** - This README file contains information about the project, including the dataset, training process, and results.

## Environment Setup

We ran the notebooks using the `virtualenv` package. To set up the virtual environment, run the following commands:

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

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