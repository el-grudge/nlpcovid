from argparse import Namespace
from Preprocessing import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from Dataset import *
import torch.optim as optim
from sklearn.model_selection import train_test_split
import scipy
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neural_network
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    args = Namespace(
        # Preprocessing
        raw_train_dataset_csv="data/HCQ_Final_Data.csv",
        seed=823,
        clean_dataset_csv='data/covid_M.csv',
        # splitting
        train_proportion=0.8,
        test_proportion=0.2,
        cutoff=0,
        # sentence encoder
        get_sentence_encoding=True,
        delete_inconsistent=False,
        module_url="https://tfhub.dev/google/universal-sentence-encoder/4",
        model_path='model_storage/sentence_encoder_model.pth',
        load_locally=True,
        # glove embeddings
        glove_filepath='data/glove.6B.100d.txt',
        # neural nets
        classifier_class='CNN',
        learning_rate=0.001,
        num_epochs=20,
        num_channels=256,
        batch_size=128,
        hidden_dim=300,
        weight=False,
        dropout_p=0.1,
        embedding_size=100,
        model_state_file='model_storage/neural_network_model.pth',
        # tfidf
        ngram_range=(1, 1),
    )
    # Set seed
    np.random.seed(args.seed)

    # PHASE 1- Preprocessing Input File
    # Read raw data
    df_train_stances = pd.read_csv(args.raw_train_dataset_csv)
    df_train_stances.columns = ['id', 'Stance', 'text']
    df_train_stances.index = df_train_stances['id']
    df_train_original = df_train_stances.copy()

    hashtag_D = clean_text(df_train_stances, args.clean_dataset_csv)

    # remove empty text
    null_after_cleaning = df_train_original.loc[df_train_stances[df_train_stances.text == ''].id]
    null_after_cleaning.to_csv('data/null_after_cleaning.csv')
    df_train_stances = df_train_stances[df_train_stances.text != '']

    # get inconsistencies
    if args.get_sentence_encoding:
        inconsistent_tweets = delete_inconsistencies(args, df_train_stances)

    # delete inconsistencies
    if args.delete_inconsistent:
        df_train_stances = df_train_stances.drop(inconsistent_tweets.index)

    # create embeddings dataset
    embeddings_dataset = []
    for _, row in df_train_stances.iterrows():
        embeddings_dataset.append([row.id, row.Stance, row.embeddings.numpy()[0]])

    dataset_embedding = pd.DataFrame(embeddings_dataset, columns=['id', 'stance', 'embeddings'])
    embedding = dataset_embedding.embeddings.apply(pd.Series)
    dataset_df = pd.concat([dataset_embedding.stance, embedding], axis=1)
    dataset_df.index = dataset_embedding.id

    # sentence encoder Data split
    X_train, X_test, y_train, y_test = train_test_split(dataset_df.loc[:, dataset_df.columns != 'stance'], dataset_df['stance'],
                                                        random_state=0, test_size=0.2)

    X_train = torch.tensor(np.matrix(X_train)).float()
    X_test = torch.tensor(np.matrix(X_test)).float()

    y_train = torch.tensor(y_train.values)
    y_test = torch.tensor(y_test.values)

    # PHASE 2- Classifications
    accuracy_scores = {"train_scores": [], "test_scores": []}
    models = []

    # Neural Networks
    # classifier_classes = ['MLP', 'CNN', 'GloVe']
    classifier_classes = ['MLP']
    for args.classifier_class in classifier_classes:

        loss_func = nn.CrossEntropyLoss()
        with_weights = [False, True]
        for args.weight in with_weights:

            if args.weight:
                loss_func.weight = get_class_weights(dataset_df)
                with_weight_str = 'weighted'
            else:
                loss_func.weight = None
                with_weight_str = 'not_weighted'

            dimensions = {
                'input_dim': X_train.shape[1],
                'hidden_dim': args.hidden_dim,
                'output_dim': dataset_df['stance'].nunique()
            }

            classifier_name = args.classifier_class + '_' + with_weight_str
            models.append(classifier_name)

            classifier = NLPClassifier(args, dimensions)

            # Forward pass, get our logits
            logps = classifier(X_train)
            # Calculate the loss with the logits and the labels
            loss = loss_func(logps, y_train)

            loss.backward()

            optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

            epoch_bar = tqdm(desc='training routine',
                             total=args.num_epochs,
                             position=0)

            train_losses = []
            test_losses = []
            test_accuracies = []

            for epoch_index in range(args.num_epochs):
                optimizer.zero_grad()

                output = classifier.forward(X_train)
                loss = loss_func(output, y_train)
                loss.backward()
                train_loss = loss.item()
                train_losses.append(train_loss)

                optimizer.step()

                # Turn off gradients     for validation, saves memory and computations
                with torch.no_grad():
                    classifier.eval()
                    log_ps = classifier(X_test)
                    test_loss = loss_func(log_ps, y_test)
                    test_losses.append(test_loss)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == y_test.view(*top_class.shape)
                    test_accuracy = torch.mean(equals.float())
                    test_accuracies.append(test_accuracy)

                classifier.train()

                print(f"Epoch: {epoch_index + 1}/{args.num_epochs}.. ",
                      f"Training Loss: {train_loss:.3f}.. ",
                      f"Test Loss: {test_loss:.3f}.. ",
                      f"Test Accuracy: {test_accuracy:.3f}")
