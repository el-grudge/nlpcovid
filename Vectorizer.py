import numpy as np
from collections import Counter
from Vocabulary import Vocabulary
from SequenceVocabulary import SequenceVocabulary


class CovidVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, predictor_vocab, target_vocab, args, max_predictor_length=0):
        """
        Args:
            predictor_vocab (Vocabulary): maps words to integers
            target_vocab (Vocabulary): maps class labels to integers
        """
        self.vectorizer_method = args.vectorizer_method
        self.predictor_vocab = predictor_vocab
        self.target_vocab = target_vocab
        self._max_predictor_length = max_predictor_length

    def vectorize(self, predictor, classifier_class):

        if classifier_class == 'MLP':
            one_hot_matrix = np.zeros(len(self.predictor_vocab), dtype=np.float32)

            for token in predictor.split(" "):
                one_hot_matrix[self.predictor_vocab.lookup_token(
                    token)] = 1 if self.vectorizer_method == 'OneHot' else self.predictor_vocab.lookup_token(token)

        elif classifier_class == 'CNN':
            if self.vectorizer_method == 'OneHot':
                one_hot_matrix_size = (len(self.predictor_vocab), self._max_predictor_length)
                one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)

                for position_index, word in enumerate(predictor.split(' ')):
                    word_index = self.predictor_vocab.lookup_token(word)
                    one_hot_matrix[word_index][position_index] = 1 if self.vectorizer_method == 'OneHot' else word_index

            elif self.vectorizer_method == 'GloVe':
                # +1 if only using begin_seq, +2 if using both begin and end seq tokens
                vector_length = self._max_predictor_length + 2
                indices = [self.predictor_vocab.begin_seq_index]
                indices.extend(self.predictor_vocab.lookup_token(token) for token in predictor.split(' '))
                indices.append(self.predictor_vocab.end_seq_index)

                if vector_length < 0:
                    vector_length = len(indices)

                one_hot_matrix = np.zeros(vector_length, dtype=np.int64)
                one_hot_matrix[:len(indices)] = indices
                one_hot_matrix[len(indices):] = self.predictor_vocab.mask_index

        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, predictor_df, args):

        """Instantiate the vectorizer from the dataset dataframe

        Args:
            predictor_df (pandas.DataFrame): the predictor dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the CovidVectorizer
        """

        predictor_vocab = SequenceVocabulary()

        target_vocab = Vocabulary(add_unk=False)
        max_predictor_length = 0

        # Add targets
        for target in sorted(set(predictor_df.Stance)):
            target_vocab.add_token(target)

        # Add top words if count > provided count
        word_counts = Counter()
        for index, row in predictor_df.iterrows():
            vector = row.text
            max_predictor_length = max(max_predictor_length, len(vector.split(' ')))
            for word in vector.split(' '):
                word_counts[word] += 1

        for word, count in word_counts.items():
            if count > args.cutoff:
                predictor_vocab.add_token(word)

        return cls(predictor_vocab, target_vocab, args, max_predictor_length)