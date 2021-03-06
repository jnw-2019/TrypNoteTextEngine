import pandas as pd
import os

# Set Pandas to display all rows of dataframes
pd.set_option('display.max_rows', 500)

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# nltk
from nltk import tokenize

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# suppress deprecation warnings
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

class LDA():
    
    def __init__(self):
        self.data = []
        self.cwd = os.getcwd()
        self.mallet_path = self.cwd + '/mallet-2.0.8/bin/mallet' # update this path

    def loaddata(self, data):
        self.data = tokenize.sent_tokenize(data)

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def compute_coherence_values(self, dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            print('Calculating {}-topic model'.format(num_topics))
            model = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)

    def process(self):

        # Tokenize into words
        print('Tokenizing')
        data_words = list(self.sent_to_words(self.data))

        # Build the bigram and trigram models
        print('Creating bigrams and trigrams')
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # Remove Stop Words
        print('Removing stopwords')
        data_words_nostops = self.remove_stopwords(data_words)

        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        def make_trigrams(texts):
            return [trigram_mod[bigram_mod[doc]] for doc in texts]

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            """https://spacy.io/api/annotation"""
            texts_out = []
            for sent in texts:
                doc = nlp(" ".join(sent)) 
                texts_out.append([token.lemma_ for token in doc])# if token.pos_ in allowed_postags])
            return texts_out

        # Form Bigrams
        print('Forming bigrams')
        data_words_bigrams = make_bigrams(data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        print('Lemmatizing')
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # Create Dictionary
        print('Creating dictionary')
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        print('Creating corpus')
        texts = data_lemmatized

        # Term Document Frequency
        print('Creating term frequency list')
        corpus = [id2word.doc2bow(text) for text in texts]

        ldamallet = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

        # Can take a long time to run.
        limit=4; start=2; step=1;
        model_list, coherence_values = self.compute_coherence_values(dictionary=id2word,
                                                                corpus=corpus,
                                                                texts=data_lemmatized,
                                                                start=start,
                                                                limit=limit,
                                                                step=step)

        # Select the model and print the topics
        index, value = max(enumerate(coherence_values), key=operator.itemgetter(1))
        #index = 17
        optimal_model = model_list[index]
        model_topics = optimal_model.show_topics(num_topics=1000, formatted=False)

        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=8, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

        # Compute Perplexity
        print ('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print ('Coherence Score: ', coherence_lda)

        df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        # Number of Documents for Each Topic
        topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts/topic_counts.sum(), 4)

        # Topic Number and Keywords
        topic_num_keywords = sent_topics_sorteddf_mallet[['Topic_Num', 'Keywords']]

        # Concatenate Column wise
        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

        # Change Column names
        df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Percent_Documents']

        # Group top 5 sentences under each topic
        sent_topics_sorteddf_mallet = pd.DataFrame()

        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], axis=0)

        # Reset Index    
        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

        # Format
        sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pd.set_option('display.max_colwidth', -1)
        
        return df_dominant_topics.to_json

    def runlda(self, data):
        print(data['postObj'])
        self.loaddata(data['postObj'])

        return self.data