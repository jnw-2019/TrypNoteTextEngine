## Sentiment Analysis ##
import pandas as pd
import numpy as np
import json
from collections import defaultdict

# Set Pandas to display all rows of dataframes
pd.set_option('display.max_rows', 500)

# nltk
from nltk import tokenize
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def runmodel(rawdata):
    text = rawdata
    # Initialize analyzer
    analyzer = SentimentIntensityAnalyzer()
    # Create sentence list
    sentence_list = tokenize.sent_tokenize(text)
    # Starting object
    sentiments = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
    # Analyze sentiment
    for sentence in sentence_list:
        #print(sentence)
        vs = analyzer.polarity_scores(sentence)
        sentiments['compound'] += vs['compound']
        sentiments['neg'] += vs['neg']
        sentiments['neu'] += vs['neu']
        sentiments['pos'] += vs['pos']
        
    sentiments['compound'] = sentiments['compound'] / len(sentence_list)
    sentiments['neg'] = sentiments['neg'] / len(sentence_list)
    sentiments['neu'] = sentiments['neu'] / len(sentence_list)
    sentiments['pos'] = sentiments['pos'] / len(sentence_list)

    results = {'sentiments': sentiments}

    def text_emotion(df, column):
        '''
        Takes a DataFrame and a specified column of text and adds 10 columns to the
        DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
        column containing the value of the text in that emotions
        INPUT: DataFrame, string
        OUTPUT: the original DataFrame with ten new columns
        '''

        new_df = df.copy()

        filepath = ('data/'
                    'NRC-Sentiment-Emotion-Lexicons/'
                    'NRC-Emotion-Lexicon-v0.92/'
                    'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
        emolex_df = pd.read_csv(filepath,
                                names=["word", "emotion", "association"],
                                sep='\t')
        emolex_words = emolex_df.pivot(index='word',
                                    columns='emotion',
                                    values='association').reset_index()
        emotions = emolex_words.columns.drop('word')
        emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

        stemmer = SnowballStemmer("english")
        
        for i, row in new_df.iterrows():

            document = word_tokenize(new_df.loc[i][column])

            for word in document:
                word = stemmer.stem(word.lower())
                emo_score = emolex_words[emolex_words.word == word]
                if not emo_score.empty:
                    for emotion in list(emotions):
                        emo_df.at[i, emotion] += emo_score[emotion]

        new_df = pd.concat([new_df, emo_df], axis=1)

        return new_df
    
    data = {'text': []}
    data['text'].append(rawdata)

    book_df = pd.DataFrame(data=data)

    book_df = text_emotion(book_df, 'text')

    print(book_df.head())

    book_df['word_count'] = book_df['text'].apply(tokenize.word_tokenize).apply(len)

    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']

    for emotion in emotions:
        book_df[emotion] = book_df[emotion] / book_df['word_count']

    print(book_df.head())

    emotionResults = book_df[emotions].to_dict()

    print(emotionResults)

    results['emotions'] = emotionResults

    return json.dumps(results)
