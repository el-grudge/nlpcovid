# nlpcovid

## Introduction

This repository contains code used for the annotation assessment in Section 3.3 of the paper "A stance data set on polarized conversations on Twitter about the efficacy of hydroxychloroquine as a treatment for COVID-19" available [here](https://www.sciencedirect.com/science/article/pii/S235234092031283X#sec0002). The repository is maintained by an acknowledged contributor to the research.

## Annotation Assessment Process

The COVID-CQ data set is created via a joint annotation of tweets' text and the shared URLs when the tweets were not self-explanatory. This approach challenges prediction models for stance detection tasks to incorporate further information besides the text of the tweets. The process involves:

- Joint Annotation: Tweets and their associated URLs are annotated together to provide a comprehensive understanding of the stance.
- Stance Detection: The data set is designed to help develop state-of-the-art models for stance classification by incorporating additional information beyond tweet text.

## Reference to the Paper

The paper discussing the COVID-CQ data set can be found here: https://www.sciencedirect.com/science/article/pii/S235234092031283X#sec0002.

## Code Structure

The repository includes several files relevant to the annotation assessment process, including data preprocessing, stance classification, and preliminary analyses.

- main.py: Reads input, calls cleaning functions, and calls classification functions.
- Preprocessing.py: Text preprocessing and cleaning functions.
- utils.py: Other utility functions.
- Dataset.py: Dataset class, creates a dataset object that has the training and test data, the vectorizer object (only for one-hot and glove), and the input vocabulary.
- Vectorizer.py: Vectorizer class, creates the vectorizer object that handles one-hot and glove vectorization. For Tf-IDF, it uses sklearn's TfidfVectorizer.
- Vocabulary.py: Vocabulary class, creates the vocabulary object that will be attached to the vectorizer object.
- SequenceVocabulary.py: Similar to the vocabulary class, but has additional tokens for text start/end markers. Used with glove vectorization.

## Results

The following barchart shows the test results of the classification models implemented here:

![alt text](./results/test_accuracies.png)

## Acknowledgement
The author of this repository is an acknowledged contributor to the research paper but not a co-author of the paper. 
