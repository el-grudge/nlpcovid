import pandas as pd
import numpy as np
import collections
from gensim.parsing.preprocessing import remove_stopwords
import inflect
import re, unicodedata
from nltk.stem.snowball import SnowballStemmer


def clean_text(covid_M, clean_dataset_csv):
    # Text preprocessing
    # lower case
    covid_M.text = covid_M.text.str.lower()

    # remove urls
    covid_M.text = covid_M.text.str.replace(r"https:\/\/t.co\/[A-Za-z0-9]+", "")

    # remove contractions
    covid_M.text = covid_M.text.str.replace(r"it['‘’]+s", "it is")
    covid_M.text = covid_M.text.str.replace(r"it\x89Ûªs", "it is")
    covid_M.text = covid_M.text.str.replace(r"it['‘’]+ll", "it will")
    covid_M.text = covid_M.text.str.replace(r"it\x89Ûªll", "it will")
    covid_M.text = covid_M.text.str.replace(r"he['‘’]+s", "he is")
    covid_M.text = covid_M.text.str.replace(r"he\x89Ûªs", "he is")
    covid_M.text = covid_M.text.str.replace(r"he['‘’]+ll", "he will")
    covid_M.text = covid_M.text.str.replace(r"he\x89Ûªll", "he will")
    covid_M.text = covid_M.text.str.replace(r"there['‘’]+s", "there is")
    covid_M.text = covid_M.text.str.replace(r"there\x89Ûªs", "there is")
    covid_M.text = covid_M.text.str.replace(r"we['‘’]+re", "we are")
    covid_M.text = covid_M.text.str.replace(r"we['‘’]+ve", "we have")
    covid_M.text = covid_M.text.str.replace(r"we['‘’]+ll", "we will")
    covid_M.text = covid_M.text.str.replace(r"we['‘’]+d", "we would")
    covid_M.text = covid_M.text.str.replace(r"we\x89Ûªre", "we are")
    covid_M.text = covid_M.text.str.replace(r"we\x89Ûªve", "we have")
    covid_M.text = covid_M.text.str.replace(r"we\x89Ûªll", "we will")
    covid_M.text = covid_M.text.str.replace(r"we\x89Ûªd", "we would")
    covid_M.text = covid_M.text.str.replace(r"that['‘’]+s", "that is")
    covid_M.text = covid_M.text.str.replace(r"that\x89Ûªs", "that is")
    covid_M.text = covid_M.text.str.replace(r"won['‘’]+t", "will not")
    covid_M.text = covid_M.text.str.replace(r"won\x89Ûªt", "will not")
    covid_M.text = covid_M.text.str.replace(r"they['‘’]+re", "they are")
    covid_M.text = covid_M.text.str.replace(r"they['‘’]+ll", "they will")
    covid_M.text = covid_M.text.str.replace(r"they['‘’]+d", "they would")
    covid_M.text = covid_M.text.str.replace(r"they['‘’]+ve", "they have")
    covid_M.text = covid_M.text.str.replace(r"they\x89Ûªre", "they are")
    covid_M.text = covid_M.text.str.replace(r"they\x89Ûªll", "they will")
    covid_M.text = covid_M.text.str.replace(r"they\x89Ûªd", "they would")
    covid_M.text = covid_M.text.str.replace(r"they\x89Ûªve", "they have")
    covid_M.text = covid_M.text.str.replace(r"can['‘’]+t", "cannot")
    covid_M.text = covid_M.text.str.replace(r"can\x89Ûªt", "cannot")
    covid_M.text = covid_M.text.str.replace(r"wasn['‘’]+t", "was not")
    covid_M.text = covid_M.text.str.replace(r"wasn\x89Ûªt", "was not")
    covid_M.text = covid_M.text.str.replace(r"don['‘’]+t", "do not")
    covid_M.text = covid_M.text.str.replace(r"don\x89Ûªt", "do not")
    covid_M.text = covid_M.text.str.replace(r"aren['‘’]+t", "are not")
    covid_M.text = covid_M.text.str.replace(r"aren\x89Ûªt", "are not")
    covid_M.text = covid_M.text.str.replace(r"isn['‘’]+t", "is not")
    covid_M.text = covid_M.text.str.replace(r"isn\x89Ûªt", "is not")
    covid_M.text = covid_M.text.str.replace(r"what['‘’]+s", "what is")
    covid_M.text = covid_M.text.str.replace(r"what\x89Ûªs", "what is")
    covid_M.text = covid_M.text.str.replace(r"haven['‘’]+t", "have not")
    covid_M.text = covid_M.text.str.replace(r"haven\x89Ûªt", "have not")
    covid_M.text = covid_M.text.str.replace(r"hasn['‘’]+t", "has not")
    covid_M.text = covid_M.text.str.replace(r"hasn\x89Ûªt", "has not")
    covid_M.text = covid_M.text.str.replace(r"you['‘’]+ve", "you have")
    covid_M.text = covid_M.text.str.replace(r"you\x89Ûªve", "you have")
    covid_M.text = covid_M.text.str.replace(r"you['‘’]+re", "you are")
    covid_M.text = covid_M.text.str.replace(r"you\x89Ûªre", "you are")
    covid_M.text = covid_M.text.str.replace(r"you['‘’]+d", "You would")
    covid_M.text = covid_M.text.str.replace(r"you\x89Ûªd", "You would")
    covid_M.text = covid_M.text.str.replace(r"you['‘’]+ll", "you will")
    covid_M.text = covid_M.text.str.replace(r"you\x89Ûªll", "you will")
    covid_M.text = covid_M.text.str.replace(r"y['‘’]+all", "you all")
    covid_M.text = covid_M.text.str.replace(r"y\x89Ûªall", "you all")
    covid_M.text = covid_M.text.str.replace(r"youve", "you have")
    covid_M.text = covid_M.text.str.replace(r"i['‘’]+m", "i am")
    covid_M.text = covid_M.text.str.replace(r"i\x89Ûªm", "i am")
    covid_M.text = covid_M.text.str.replace(r"shouldn['‘’]+t", "should not")
    covid_M.text = covid_M.text.str.replace(r"shouldn\x89Ûªt", "should not")
    covid_M.text = covid_M.text.str.replace(r"wouldn['‘’]+t", "would not")
    covid_M.text = covid_M.text.str.replace(r"wouldn\x89Ûªt", "would not")
    covid_M.text = covid_M.text.str.replace(r"would['‘’]+ve", "would have")
    covid_M.text = covid_M.text.str.replace(r"would\x89Ûªve", "would have")
    covid_M.text = covid_M.text.str.replace(r"here['‘’]+s", "here is")
    covid_M.text = covid_M.text.str.replace(r"here\x89Ûªs", "here is")
    covid_M.text = covid_M.text.str.replace(r"couldn['‘’]+t", "could not")
    covid_M.text = covid_M.text.str.replace(r"couldn\x89Ûªt", "could not")
    covid_M.text = covid_M.text.str.replace(r"doesn['‘’]+t", "does not")
    covid_M.text = covid_M.text.str.replace(r"doesn\x89Ûªt", "does not")
    covid_M.text = covid_M.text.str.replace(r"who['‘’]+s", "who is")
    covid_M.text = covid_M.text.str.replace(r"who\x89Ûªs", "who is")
    covid_M.text = covid_M.text.str.replace(r"i['‘’]+ve", "i have")
    covid_M.text = covid_M.text.str.replace(r"i\x89Ûªve", "i have")
    covid_M.text = covid_M.text.str.replace(r"weren['‘’]+t", "were not")
    covid_M.text = covid_M.text.str.replace(r"weren\x89Ûªt", "were not")
    covid_M.text = covid_M.text.str.replace(r"didn['‘’]+t", "did not")
    covid_M.text = covid_M.text.str.replace(r"didn\x89Ûªt", "did not")
    covid_M.text = covid_M.text.str.replace(r"i['‘’]+d", "i would")
    covid_M.text = covid_M.text.str.replace(r"I\x89Ûªd", "i would")
    covid_M.text = covid_M.text.str.replace(r"should['‘’]+ve", "should have")
    covid_M.text = covid_M.text.str.replace(r"should\x89Ûªve", "should have")
    covid_M.text = covid_M.text.str.replace(r"where['‘’]+s", "where is")
    covid_M.text = covid_M.text.str.replace(r"where\x89Ûªs", "where is")
    covid_M.text = covid_M.text.str.replace(r"i['‘’]+ll", "i will")
    covid_M.text = covid_M.text.str.replace(r"i\x89Ûªll", "i will")
    covid_M.text = covid_M.text.str.replace(r"let['‘’]+s", "let us")
    covid_M.text = covid_M.text.str.replace(r"let\x89Ûªs", "let us")
    covid_M.text = covid_M.text.str.replace(r"ain['‘’]+t", "am not")
    covid_M.text = covid_M.text.str.replace(r"ain\x89Ûªt", "am not")
    covid_M.text = covid_M.text.str.replace(r"let['‘’]+s", "let us")
    covid_M.text = covid_M.text.str.replace(r"let\x89Ûªs", "let us")
    covid_M.text = covid_M.text.str.replace(r"ain['‘’]+t", "am not")
    covid_M.text = covid_M.text.str.replace(r"ain\x89Ûªt", "am not")
    covid_M.text = covid_M.text.str.replace(r"could['‘’]+ve", "could have")
    covid_M.text = covid_M.text.str.replace(r"could\x89Ûªve", "could have")
    covid_M.text = covid_M.text.str.replace(r"donå«t", "do not")
    covid_M.text = covid_M.text.str.replace(r"vit c", "vitamin c")
    covid_M.text = covid_M.text.str.replace(r"ź", "z")
    covid_M.text = covid_M.text.str.replace(r"\\n", " ")
    # potentials: mx-max, drs-doctors, dr-doctor

    # remove acronyms
    covid_M.text = covid_M.text.str.replace(r"u\.s\.", "united states")
    covid_M.text = covid_M.text.str.replace(r"n\.y\.", "new york")

    # fix character entity references
    covid_M.text = covid_M.text.str.replace(r"&amp;", "and")
    covid_M.text = covid_M.text.str.replace(r"&gt;", "greater than")
    covid_M.text = covid_M.text.str.replace(r"&lt;", "less than")

    # removing mentions
    covid_M.text = covid_M.text.str.replace(r"@[A-Za-z0-9]+", "")

    # handle padding punctuation and special characters
    covid_M.text = covid_M.text.str.replace(r"([!?+*\[\]\-%:/();$=><|{}^'`ー‘’.\"_,@—“”\\])", r" \1 ")

    # removing punctuation and special characters
    covid_M.text = covid_M.text.str.replace(r"([!?+*\[\]\-%:/();$=><|{}^'`ー‘’.\"_,@—“”\\])", r"")

    # removing multiple spaces
    covid_M.text = covid_M.text.str.replace(r"\s+", r" ")
    covid_M.text = covid_M.text.str.replace(r"^\s", '')

    # hashtag dictionary
    hashtags = covid_M.text.str.findall(r"#[\w\.-]+")
    hashtags = [x for x in hashtags if x != []]
    hashtags = set(item for sublist in hashtags for item in sublist)

    by_hashtag = collections.defaultdict(list)
    for hashtag in hashtags:
        covid_M_hashtag = pd.Series(covid_M.text.str.find(hashtag))
        by_hashtag[hashtag].append(covid_M.loc[covid_M_hashtag[covid_M_hashtag > -1].index.values.tolist()]['text'])

    hashtag_D = pd.DataFrame([(key, len(by_hashtag[key][0])) for key in list(by_hashtag.keys())],
                             columns=['hashtag', 'count']).sort_values(by='count', ascending=False).reset_index(
        drop=True)

    # remove hashtags
    covid_M.text = covid_M.text.str.replace(r"#[\w\.-]+", '')

    # stopwords
    covid_M.text = covid_M.text.apply(remove_stopwords)

    # remove rows with no text
    covid_M['text'].replace('', np.nan, inplace=True)
    covid_M.dropna(subset=['text'], inplace=True)

    def remove_non_ascii(words_V):
        """Remove non-ASCII characters from list of tokenized words"""
        str = ''
        for word in words_V:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            str += new_word
        return str

    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        str = ''
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                str += new_word
        return str

    def replace_numbers(words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        str = ''
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                str += new_word
            else:
                str += word
        return str

    def stemming(words):
        str = ''
        stemmer = SnowballStemmer("english")

        for word in words:
            stem = stemmer.stem(word)
            str += stem
        return str

    def normalize_tweets(words):
        words = remove_non_ascii(words)
        words = remove_punctuation(words)
        words = replace_numbers(words)
        #   words = stemming(words)
        return words

    covid_M.text = covid_M.text.apply(normalize_tweets)

    # write to file
    covid_M.to_csv(clean_dataset_csv, index=False)

    return hashtag_D
