import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from CNN import *
from MLP import *
from GloVeClassifier import *
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub
import collections


def save_misclassified(text, y_test, predicted, output_file):
    misclassified = pd.DataFrame(text.loc[y_test[y_test != predicted].index])
    misclassified['y_test'] = y_test[y_test != predicted]
    misclassified['predicted'] = predicted[y_test != predicted]
    misclassified.to_csv(Path().joinpath('misclassified', output_file))


def compute_accuracy(y_pred, y_target):
    """Predict the target of a predictor"""
    y_pred_indices = y_pred.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def generate_batches(dataset, batch_size, shuffle=False, drop_last=True):
    """
        A generator function which wraps the PyTorch DataLoader. It will
          ensure each tensor is on the write device location.
        """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name]
        yield out_data_dict

'''
def NLPClassifier(args, dimensions):
    """Builds a classifier

    Args:
        args: main arguments
        classifier_class: classifier class to be defined
        dimensions: neural network dimensions
        loss_func: loss function to be used

    Returns:
        classifier: built classfier
        loss_func: loss function
        optimizer: optimizer
        scheduler
    """
    if args.classifier_class == 'MLP':

        
        classifier = MLPClassifier(input_dim=dimensions['input_dim'],
                                   hidden_dim=dimensions['hidden_dim'],
                                   output_dim=dimensions['output_dim'])
        
        classifier = MLPClassifier(input_dim=dimensions['input_dim'],
                                   hidden_dim=dimensions['hidden_dim'],
                                   output_dim=dimensions['output_dim'],
                                   embedding_size=args.embedding_size,
                                   num_embeddings=dimensions['input_dim'],
                                   dropout_p=args.dropout_p,
                                   pretrained_embeddings=dimensions['pretrained_embeddings'])


    elif args.classifier_class == 'CNN':
        classifier = CNNClassifier(initial_num_channels=dimensions['input_dim'],
                                   num_classes=dimensions['output_dim'],
                                   num_channels=args.num_channels)

    # GLOVE_MODEL
    elif args.classifier_class == 'GloVe':
        classifier = GloVeClassifier(embedding_size=args.embedding_size,
                                     num_embeddings=dimensions['input_dim'],
                                     num_channels=args.num_channels,
                                     hidden_dim=args.hidden_dim,
                                     num_classes=dimensions['output_dim'],
                                     dropout_p=args.dropout_p,
                                     pretrained_embeddings=dimensions['pretrained_embeddings'],
                                     padding_idx=dimensions['padding_idx'])

    return classifier
    '''


def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings


# GLOVE_MODEL
def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings

    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def predict_target(args, predictor, classifier, vectorizer):
    """Predict the target of a predictor for Perceptron

        Args:
            predictor (str): the text of the predictor
            classifier (Perceptron): the trained model
            vectorizer (ReviewVectorizer): the corresponding vectorizer
            decision_threshold (float): The numerical boundary which separates the target classes
            :param classifier_class: classifier class
    """
    if args.classifier_class == 'MLP':
        vectorized_predictor = torch.tensor(vectorizer.vectorize(predictor, args.classifier_class)).view(1, -1)
    else:
        vectorized_predictor = torch.tensor(vectorizer.vectorize(predictor, args.classifier_class)).unsqueeze(0)

    result = classifier(vectorized_predictor, apply_softmax=True)
    probability_values, indices = result.max(dim=1)
    index = indices.item()

    return vectorizer.target_vocab.lookup_index(index)


def delete_inconsistencies(args, covid_M):
    # sentence encoding
    if(args.load_locally):
        sentence_encoder = tf.saved_model.load(args.model_path)
    else:
        sentence_encoder = hub.load(args.module_url)

    tf.saved_model.save(sentence_encoder, args.model_path)
    print("module %s loaded" % args.module_url)

    covid_M['embeddings'] = [sentence_encoder([tweet]) for tweet in covid_M.text]

    # Cosine distance
    cos_dis = pd.DataFrame(data=cosine_similarity(
        pd.DataFrame(np.concatenate([np.array(embedding) for embedding in covid_M.embeddings]),
                     index=covid_M.index)), columns=covid_M.index, index=covid_M.index)

    # identify duplicate tweets
    similar_tweets = collections.defaultdict(list)
    [similar_tweets[cos].append(cos_dis[cos_dis[cos] > 0.9].index.tolist()) for cos in cos_dis]
    [similar_tweets[k][0].remove(k) for k in similar_tweets.keys()]
    similar_tweets_keys = list(similar_tweets.keys())
    for k in similar_tweets_keys:
        if len(similar_tweets[k][0])==0:
            del similar_tweets[k]
        else:
            similar_tweets[k] = similar_tweets[k][0]

    inconsistent_tweets = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
    for key in similar_tweets.keys():
        for value in similar_tweets[key]:
            if covid_M.loc[key, 'Stance'] != covid_M.loc[value, 'Stance']:
                inconsistent_tweets = inconsistent_tweets.append(pd.DataFrame(covid_M.loc[[key], ['Stance','text']]))
                inconsistent_tweets = inconsistent_tweets.append(pd.DataFrame(covid_M.loc[[value], ['Stance','text']]))

    inconsistent_tweets.to_csv('data/inconsistent_tweets.csv', sep='\t')

    return inconsistent_tweets


def get_class_weights(predictor_df):
    # Class weights
    class_counts = predictor_df.stance.value_counts().to_dict()

    sorted_counts = sorted(class_counts.items())
    frequencies = [count for _, count in sorted_counts]
    class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    return class_weights



def NLPClassifier(args, dimensions):
    """Builds a classifier

    Args:
        args: main arguments
        classifier_class: classifier class to be defined
        dimensions: neural network dimensions
        loss_func: loss function to be used

    Returns:
        classifier: built classfier
        loss_func: loss function
        optimizer: optimizer
        scheduler
    """
    if args.classifier_class == 'MLP':
        classifier = MLPClassifier(input_dim=dimensions['input_dim'],
                                   hidden_dim=dimensions['hidden_dim'],
                                   output_dim=dimensions['output_dim'],
                                   embedding_size=args.embedding_size,
                                   num_embeddings=dimensions['input_dim'],
                                   dropout_p=args.dropout_p,
                                   pretrained_embeddings=dimensions['pretrained_embeddings'],
                                   padding_idx=dimensions['padding_idx'],
                                   vectorizer_method=args.vectorizer_method)

    elif args.classifier_class == 'CNN':
        if args.vectorizer_method == 'OneHot':
            classifier = CNNClassifier(initial_num_channels=dimensions['input_dim'],
                                       hidden_dim=args.hidden_dim,
                                       num_classes=dimensions['output_dim'],
                                       num_channels=args.num_channels,
                                       dropout_p=args.dropout_p)

        elif args.vectorizer_method == 'GloVe':
            # GLOVE_MODEL
            classifier = GloVeClassifier(embedding_size=args.embedding_size,
                                         num_embeddings=dimensions['input_dim'],
                                         num_channels=args.num_channels,
                                         hidden_dim=args.hidden_dim,
                                         num_classes=dimensions['output_dim'],
                                         dropout_p=args.dropout_p,
                                         pretrained_embeddings=dimensions['pretrained_embeddings'],
                                         padding_idx=dimensions['padding_idx'])

    return classifier
