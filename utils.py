import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from CNN import *
from MLP import *
from GloVeClassifier import *
from pathlib import Path


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
