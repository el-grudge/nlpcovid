from argparse import Namespace
from Preprocessing import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from Dataset import *
import torch.optim as optim
from sklearn.model_selection import train_test_split
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

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(df_train_stances['text'], df_train_stances['Stance'],
                                                        random_state=0, test_size=0.2)
    train = pd.concat([X_train, y_train], axis=1)
    train['split'] = 'train'
    test = pd.concat([X_test, y_test], axis=1)
    test['split'] = 'test'
    final_predictors = pd.concat([train, test])
    final_predictors['id'] = final_predictors.index

    # Fill the 'split' value for rows with null 'split' value
    for i in final_predictors[pd.isnull(final_predictors.split)].index:
        final_predictors.loc[i, 'split'] = final_predictors.loc[i - 1, 'split']

    # Convert target value from integer to string
    final_predictors['Stance'] = final_predictors.Stance.astype(str)

    # Write preprocessed dataset to csv file
    final_predictors.to_csv(args.clean_dataset_csv, index=False)

    # PHASE 2- Classifications
    accuracy_scores = {"train_scores": [], "test_scores": []}
    models = []

    # Neural Networks
    classifier_classes = ['MLP', 'CNN', 'GloVe']
    for args.classifier_class in classifier_classes:

        dataset = CovidDataset.load_dataset_and_make_vectorizer(args)

        # Neural Networks
        vectorizer = dataset.get_vectorizer()

        loss_func = nn.CrossEntropyLoss()
        with_weights = [False, True]
        for args.weight in with_weights:
            if args.weight:
                loss_func.weight = dataset.class_weights
                with_weight_str = 'weighted'
            else:
                loss_func.weight = None
                with_weight_str = 'not_weighted'

            # GLOVE_MODEL
            # Use GloVe or randomly initialized embeddings
            if args.classifier_class == 'GloVe':
                vectorizer_method = 'GloVe'
                words = vectorizer.predictor_vocab._token_to_idx.keys()
                embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
                                                   words=words)
                print("Using pre-trained embeddings")
            else:
                vectorizer_method = 'OneHot'
                print("Not using pre-trained embeddings")
                embeddings = None

            dimensions = {
                'input_dim': len(vectorizer.predictor_vocab),
                'hidden_dim': args.hidden_dim,
                'output_dim': len(vectorizer.target_vocab),
                'dropout_p': args.dropout_p,  # GLOVE_MODEL
                'pretrained_embeddings': embeddings,  # GLOVE_MODEL
                'padding_idx': 0  # GLOVE_MODEL
            }

            classifier_name = args.classifier_class + '_' + vectorizer_method + '_' + with_weight_str
            models.append(classifier_name)

            classifier = NLPClassifier(args, dimensions)
            optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

            epoch_bar = tqdm(desc='training routine',
                             total=args.num_epochs,
                             position=0)

            for epoch_index in range(args.num_epochs):

                dataset.set_split('train')
                train_bar = tqdm(desc='split=train',
                                 total=dataset.get_num_batches(args.batch_size),
                                 position=1,
                                 leave=True)

                # setup: batch generator, set loss and acc to 0, set train mode on
                batch_generator = generate_batches(dataset, batch_size=args.batch_size)
                running_loss = 0.0
                running_acc = 0.0
                classifier.train()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # the training routine is these 5 steps:
                    # --------------------------------------
                    # step 1. zero the gradients
                    optimizer.zero_grad()

                    # step 2. compute the output
                    y_pred = classifier(batch_dict['x_data'])

                    # step 3. compute the loss
                    loss = loss_func(y_pred, batch_dict['y_target'])
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # step 4. use loss to produce gradients
                    loss.backward()

                    # step 5. use optimizer to take gradient step
                    optimizer.step()
                    # -----------------------------------------
                    # compute the accuracy
                    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    # update bar
                    train_bar.set_postfix(loss=running_loss,
                                          acc=running_acc,
                                          epoch=epoch_index)
                    train_bar.update()

                torch.save(classifier.state_dict(), args.model_state_file)

                epoch_bar.update()

            # Test
            classifier.load_state_dict(torch.load(args.model_state_file))

            test_predictor = dataset.test_df

            text = dataset.predictor_df['text']
            text.index = dataset.predictor_df['id']

            results = []
            for _, value in test_predictor.iterrows():
                prediction = predict_target(args,
                                            value['text'],
                                            classifier,
                                            vectorizer)
                results.append([value['id'], prediction, value['Stance']])

            results = pd.DataFrame(results, columns=['id', 'predicted', 'Stance'])
            results.index = results.id
            test_accuracy = pd.Series.eq(results.predicted, results.Stance).sum().item() / results.shape[0]
            accuracy_scores['train_scores'].append(running_acc / 100)
            accuracy_scores['test_scores'].append(test_accuracy)

            save_misclassified(text, results.Stance, results.predicted, '{}_misclassified'.format(classifier_name))
            print('Accuracy of {} classifier on training set: {:.4f}'.format(classifier_name, running_acc / 100))
            print('Accuracy of {} classifier on test set: {:.4f}'.format(classifier_name, test_accuracy))
            print('Classification report')
            print(metrics.classification_report(results.Stance, results.predicted,
                                                target_names=df_train_stances['Stance'].astype(str).unique()))
            print('Confusion matrix')
            print(metrics.confusion_matrix(results.Stance, results.predicted))

    # sklearn Classifiers
    # Tf-IDF Vectorizationos

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', lowercase=True,
                                       ngram_range=args.ngram_range)
    X_train = tfidf_vectorizer.fit_transform(dataset.train_df['text'])
    y_train = dataset.train_df['Stance']
    y_train.index = dataset.train_df['id']

    X_test = tfidf_vectorizer.transform(dataset.test_df['text'])
    y_test = dataset.test_df['Stance']
    y_test.index = dataset.test_df['id']

    classifiers = [MultinomialNB(),
                   LogisticRegression(),
                   svm.SVC(kernel='linear', C=1.0),
                   SGDClassifier(loss='perceptron', penalty='l2', n_jobs=-1, max_iter=1000, warm_start=True,
                                 verbose=False),
                   GradientBoostingClassifier(verbose=False),
                   neural_network.MLPClassifier(verbose=False)]

    # classifiers = [MultinomialNB()]

    ngram_range = [(1, 1), (2, 2)]

    for args.ngram_range in ngram_range:

        if args.ngram_range == (1, 1):
            gram_range = 'unigram'
        else:
            gram_range = 'bigram'

        for clf in classifiers:
            classifier_name = ''.join(x for x in str(type(clf)).split('.')[-1] if x.isalpha())
            classifier_name = classifier_name + '_' + gram_range
            models.append(classifier_name)
            clf = clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)

            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            accuracy_scores['train_scores'].append(train_score)
            accuracy_scores['test_scores'].append(test_score)

            save_misclassified(text, y_test, predicted, '{}_misclassified'.format(classifier_name))
            print('Accuracy of {} classifier on training set: {:.4f}'.format(classifier_name, train_score))
            print('Accuracy of {} classifier on test set: {:.4f}'.format(classifier_name, test_score))
            print('Classification report')
            print(metrics.classification_report(y_test, predicted,
                                                target_names=df_train_stances['Stance'].astype(str).unique()))
            print('Confusion matrix')
            print(metrics.confusion_matrix(y_test, predicted))

    # Plotting results
    results = pd.DataFrame([accuracy_scores['train_scores'], accuracy_scores['test_scores']],
                           index=['train_scores', 'test_scores'], columns=models)
    results.transpose().to_csv('results/score_report.csv')

    for title in accuracy_scores.keys():
        fig = plt.figure()
        plt.bar(models, accuracy_scores[title], align='center')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.xticks(range(len(models)), models, size='small', rotation=45)
        plt.show()

    print('The End')
