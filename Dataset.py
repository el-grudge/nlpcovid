import pandas as pd
from torch.utils.data import Dataset
from Vectorizer import CovidVectorizer
import torch


class CovidDataset(Dataset):
    def __init__(self, predictor_df, vectorizer, classifier_class):
        """
        Args:
            predictor_df (pandas.DataFrame): the dataset
            vectorizer (CovidVectorizer): vectorizer instantiated from dataset
        """
        self.predictor_df = predictor_df
        self._vectorizer = vectorizer

        self.train_df = self.predictor_df[self.predictor_df.split == 'train']
        self.train_size = len(self.train_df)

        self.test_df = self.predictor_df[self.predictor_df.split == 'test']
        self.test_size = len(self.test_df)

        self.classifier_class = classifier_class

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        # GLOVE_MODEL
        # Class weights
        class_counts = predictor_df.Stance.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.target_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, args):
        """Load dataset and make a new vectorizer from scratch

        Args:
            predictor_csv (str): location of the dataset
        Returns:
            an instance of ReviewDataset
        """
        predictor_df = pd.read_csv(args.clean_dataset_csv)
        train_predictor_df = predictor_df[predictor_df.split == 'train']
        classifier_class = args.classifier_class
        return cls(predictor_df, CovidVectorizer.from_dataframe(train_predictor_df, args), classifier_class)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe

        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        predictor_vector = self._vectorizer.vectorize(row.text, self.classifier_class)

        target_index = self._vectorizer.target_vocab.lookup_token(row.Stance)

        return {'x_data': predictor_vector,
                'y_target': target_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

    def update_vectorizer(self, vectorizer_method):
        """Update vectorizer method

                Args:
                    vectorizer_method
        """
        self._vectorizer.vectorizer_method = vectorizer_method
